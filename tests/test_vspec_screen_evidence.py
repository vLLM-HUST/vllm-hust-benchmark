"""Recompute the local-argmax result and validate captured execution evidence."""

import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

import pytest

EVIDENCE = (
    Path(__file__).resolve().parents[1]
    / "reports/vllm-hust-midterm-20261019/vspec-school028"
)


def test_vspec_pair_reproducible():
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    assert summary["quality_rescored"] is False
    assert summary["independent_pairs"] == 3 and len(summary["runs"]) == 6
    with tarfile.open(EVIDENCE / "verification-tools.tar.gz") as tar:
        assert (
            hashlib.sha256(
                tar.extractfile("verify_school028_run.py").read()
            ).hexdigest()
            == summary["verifier_sha256"]
        )
    hashes = []
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
            hashes.append(receipt["input_hashes"])
            assert receipt["status"] == "measured" and receipt["all_devices_released"]
            assert not receipt.get("startup_foreign_context_observations")
            assert receipt["child_exit_codes"] in ([0, 0], [0, 130])
            assert raw["completed"] == 200 and raw["failed"] == 0
            assert (
                raw["total_input_tokens"] == 282265
                and raw["total_output_tokens"] == 51200
            )
            assert len(raw["output_lens"]) == 200 and set(raw["output_lens"]) == {256}
            assert raw["total_output_tokens"] / raw["duration"] == pytest.approx(
                row["metrics"]["output_throughput"]
            )
            tpot = [
                (latency - ttft) * 1000 / (length - 1)
                for latency, ttft, length in zip(
                    raw["latencies"], raw["ttfts"], raw["output_lens"]
                )
            ]
            assert statistics.mean(tpot) == pytest.approx(
                row["metrics"]["mean_tpot_ms"]
            )
            assert statistics.mean(raw["ttfts"]) * 1000 == pytest.approx(
                row["metrics"]["mean_ttft_ms"]
            )
            for name, digest in receipt["input_hashes"].items():
                assert hashlib.sha256(read(name)).hexdigest() == digest
            ranks = set()
            for name in tar.getnames():
                if not name.startswith(f"{run}/vspec-") or not name.endswith(".jsonl"):
                    continue
                events = [json.loads(line) for line in read(name).splitlines()]
                loaded = [e for e in events if e["event"] == "model_loaded"]
                assert len(loaded) == 1 and loaded[0]["active"] == (row["mode"] == "on")
                ranks.add(loaded[0]["rank"])
                checks = [e for e in events if e["event"] == "head_equivalence"]
                assert {(e["rows"], e["kind"]) for e in checks} == {
                    (n, k) for n in (1, 24, 36, 48) for k in ("zeros", "sin")
                }
                assert all(e["exact"] for e in checks)
                assert any(
                    e["event"] == "selector_executed" and e["capturing"] for e in events
                )
                assert any(
                    e["event"] == "draft_graph_replay"
                    and receipt["bench_started_wall"]
                    <= e["time"]
                    <= receipt["bench_ended_wall"]
                    for e in events
                )
            assert ranks == {0, 1}
    assert all(h == hashes[0] for h in hashes)
    for mode in ("off", "on"):
        for key, aggregate in summary["aggregates"][mode].items():
            values = [r["metrics"][key] for r in summary["runs"] if r["mode"] == mode]
            assert aggregate == dict(
                median=statistics.median(values), min=min(values), max=max(values)
            )
    values = [r["metrics"]["output_throughput"] for r in summary["runs"]]
    ratios = [values[i + 1] / values[i] for i in (0, 2, 4)]
    gain = 100 * (math.exp(statistics.mean(math.log(r) for r in ratios)) - 1)
    assert summary["on_vs_off_throughput_pct"] == pytest.approx(gain)
    assert summary["paired_throughput_ratios"] == pytest.approx(ratios)
    assert summary["promote_throughput_candidate"] == (
        statistics.median(values[1::2])
        > summary["current_best_per_metric"]["output_throughput"]
        and gain > 0
    )
