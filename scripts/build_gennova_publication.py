"""Derive the independent-engine display from sealed validation evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "reports/gennova-mmlu-pro-20261023"
TARGET = ROOT / "website-publication/data/gennova-validation-v1.json"
REVISION = "b80727393ed07a12a4dfbb5d8006c7ac8c221346"  # pragma: allowlist secret


def build():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    row = next(r for r in manifest["archives"] if r["run"] == "engine-r1")
    archive = EVIDENCE / row["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == row["sha256"]
    with tarfile.open(archive) as stream:

        def read(name):
            name = "engine-r1/" + name
            raw = stream.extractfile(name).read()
            assert hashlib.sha256(raw).hexdigest() == row["members"][name]
            return json.loads(raw)

        receipt = read("receipt.json")
        ranks = [read(f"rank-{i}.json") for i in (0, 1)]
    assert receipt["exit_code"] == 0 and receipt["all_devices_released"]
    assert receipt["source_commit"] == manifest["source_commit"]
    assert ranks[0]["outputs"] == ranks[1]["outputs"]
    assert all(len(tokens) == 256 for tokens in ranks[0]["outputs"].values())
    assert len(ranks[0]["outputs"]) == 200
    assert all(
        r["total_output_tokens"] == 51200 and r["total_input_tokens"] == 282265
        for r in ranks
    )
    elapsed = max(r["seconds"] for r in ranks)
    events = {request: [] for request in ranks[0]["outputs"]}
    for event in ranks[0]["token_delivery_events"]:
        events[event["request_id"]].append(event)
    assert all(sum(e["count"] for e in rows) == 256 for rows in events.values())
    base = f"https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/{REVISION}/reports/gennova-mmlu-pro-20261023/"
    return {
        "schema": "gennova-independent-validation/v1",
        "engine": "GenNova",
        "kind": "independent-engine",
        "status": "validated-scoped",
        "date": "2026-10-23",
        "model": "Qwen3.5-35B-A3B",
        "precision": "BF16",
        "hardware": "Ascend 910B2",
        "topology": "TP2 / PP1 / DP1",
        "capacity": ranks[0]["capacity"],
        "mtp_draft_tokens": 2,
        "source_commit": receipt["source_commit"],
        "benchmark_evidence_commit": REVISION,
        "timing_boundary": ranks[0]["timing_boundary"],
        "metrics": {
            "completed_requests": 200,
            "total_input_tokens": 282265,
            "total_output_tokens": 51200,
            "seconds": elapsed,
            "output_tokens_per_second": 51200 / elapsed,
            "mean_ttft_seconds": statistics.mean(
                rows[0]["seconds"] for rows in events.values()
            ),
            "mean_tpot_ms": statistics.mean(
                (rows[-1]["seconds"] - rows[0]["seconds"]) / 255 * 1000
                for rows in events.values()
            ),
            "rank_output_agreement": True,
            "exit_code": 0,
            "all_devices_released": True,
        },
        "full_dataset_quality_score": None,
        "b1_replacement_approved": False,
        "http_comparable": False,
        "evidence": {
            "report": base + "README.md",
            "manifest": base + "manifest.json",
            "raw_run": base + "engine-r1.tar.gz",
            "protocol": base + "protocol-and-tools.tar.gz",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    raw = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        assert TARGET.read_text() == raw, "Stale GenNova publication"
    else:
        TARGET.write_text(raw)
