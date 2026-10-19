"""Recompute the bounded Microflow screen from its sealed request records."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

import pytest


EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/microflow-school028"
)


def test_microflow_pair_is_reproducible():
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    assert summary["quality_rescored"] is False
    assert summary["independent_pairs"] == 3
    assert len(summary["runs"]) == 6
    with tarfile.open(EVIDENCE / "verification-tools.tar.gz") as tar:
        assert (
            hashlib.sha256(tar.extractfile("verify_school_run.py").read()).hexdigest()
            == summary["verifier_sha256"]
        )
    assert (
        summary["input_sha256"]
        == (
            "fb79b6fa98f232282a948e4a17f404f80c03dc2bae05af57b95b9cc0651b3506"  # pragma: allowlist secret -- public input SHA256
        )
    )
    receipts = []
    for row in summary["runs"]:
        archive = EVIDENCE / row["archive"]
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["archive_sha256"]
        run = row["verification"]["run_id"]
        with tarfile.open(archive) as tar:

            def read(name):
                return tar.extractfile(name).read()

            raw_bytes = read(f"{run}/mmlu-pro.json")
            assert hashlib.sha256(raw_bytes).hexdigest() == row["raw_sha256"]
            raw = json.loads(raw_bytes)
            receipt = json.loads(read(f"{run}/receipt.json"))
            receipts.append(receipt)
            assert receipt["status"] == "measured"
            assert receipt["all_devices_released"] is True
            assert receipt["child_exit_codes"][0] == 0
            assert raw["completed"] == 200 and raw["failed"] == 0
            assert raw["total_input_tokens"] == 282265
            assert raw["total_output_tokens"] == 51200
            assert set(raw["output_lens"]) == {256}
            assert raw["total_output_tokens"] / raw["duration"] == pytest.approx(
                row["metrics"]["output_throughput"], rel=1e-12
            )
            for name, digest in receipt["input_hashes"].items():
                assert hashlib.sha256(read(name)).hexdigest() == digest
            cache = json.loads(read(f"{run}/compiler-cache-initialization.json"))
            assert (
                cache["seed_manifest_sha256"]
                == receipt["input_hashes"]["microflow-cache-seed.json"]
            )
            assert all(
                item["time"] < receipt["bench_started_wall"]
                and item["phase"] == "starting"
                for item in receipt.get("startup_foreign_context_observations", [])
            )
            ranks = set()
            for name in tar.getnames():
                if not name.startswith(f"{run}/microflow-") or not name.endswith(
                    ".jsonl"
                ):
                    continue
                events = [json.loads(line) for line in read(name).splitlines()]
                loaded = [e for e in events if e["event"] == "model_loaded"]
                assert len(loaded) == 1
                ranks.add(loaded[0]["rank"])
                assert loaded[0]["active"] == (row["mode"] == "on")
                if row["mode"] == "on":
                    checks = [
                        e for e in events if e["event"] == "partition_equivalence"
                    ]
                    assert {e["tokens"] for e in checks} == {1024, 2048, 4096}
                    assert all(all(e["exact"]) for e in checks)
                    captures = {
                        e["layer"]
                        for e in events
                        if e["event"] == "prefix" and e["capturing"] and e["split"]
                    }
                    assert len(captures) == 10
                    assert any(
                        e["event"] == "graph_replay"
                        and e.get("target")
                        and not e["draft"]
                        and e["tokens"] in (1024, 2048, 4096)
                        and receipt["bench_started_wall"]
                        <= e["time"]
                        <= receipt["bench_ended_wall"]
                        for e in events
                    )
            assert ranks == {0, 1}
    assert all(receipts[0]["input_hashes"] == r["input_hashes"] for r in receipts)
    values = [row["metrics"]["output_throughput"] for row in summary["runs"]]
    ratios = [values[i + 1] / values[i] for i in range(0, 6, 2)]
    assert summary["paired_throughput_ratios"] == pytest.approx(ratios)
    paired_gain = 100 * (math.exp(statistics.mean(math.log(r) for r in ratios)) - 1)
    assert summary["on_vs_off_throughput_pct"] == pytest.approx(paired_gain)
    on = statistics.median(values[1::2])
    reference = summary["current_throughput_b1_metrics"]["output_throughput"]
    assert summary["on_vs_current_throughput_b1_pct"] == pytest.approx(
        100 * (on / reference - 1)
    )
    assert summary["promote_throughput_candidate"] == (
        on > reference and paired_gain > 0
    )
    for mode in ("off", "on"):
        for key, measured in summary["aggregates"][mode].items():
            data = [
                row["metrics"][key] for row in summary["runs"] if row["mode"] == mode
            ]
            assert measured == {
                "median": statistics.median(data),
                "min": min(data),
                "max": max(data),
            }
