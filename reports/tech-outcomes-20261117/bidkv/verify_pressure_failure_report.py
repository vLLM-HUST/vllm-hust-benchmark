"""Bind the historical failure report to original public records and custody."""

import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_bytes())


def verify(root):
    folder = root / "pressure-failure-report-v1"
    for item in read(folder / "manifest.json")["files"]:
        stored = (
            "original-REPORT.md.source" if item["path"] == "REPORT.md" else item["path"]
        )
        raw = (folder / stored).read_bytes()
        assert len(raw) == item["bytes"]
        assert hashlib.sha256(raw).hexdigest() == item["sha256"]
    summary = read(folder / "summary.json")
    for name, expected in summary["source_files"].items():
        directory, filename = name.split("/", 1)
        assert directory.startswith("bidkv81-")
        raw = (root / directory.removeprefix("bidkv81-") / filename).read_bytes()
        assert len(raw) == expected["bytes"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]
    planned = read(root / "pressure-v2-failed-results-v1/requests.json")
    assert len(planned) == 16
    native = [row for row in planned if row["arm"] == "native"]
    assert len(native) == 8 and all(row["status"] == "ok" for row in native)
    output = sum(row["usage"]["completion_tokens"] for row in native)
    wall = max(row["end"]["monotonic"] for row in native) - min(
        row["start"]["monotonic"] for row in native
    )
    assert output == summary["v2"]["native_accepted_output_tokens"] == 32768
    assert math.isclose(
        output / wall,
        summary["v2"]["native_metrics"]["output_tokens_per_second_if_complete"],
    )
    assert summary["v2"]["actual_controller_returncode"] == 2
    assert summary["v3"]["candidate_throughput"] is None
    assert summary["v3"]["candidate_successful"] == 0
    assert summary["v3"]["candidate_final_usage_unknown"] == 8
    preemptions = read(root / "full-exact-preemptions.json")
    assert preemptions["model_requests"] == 0
    assert len(preemptions["rows"]) == 10
    assert {(row["arm"], row["dataset"]) for row in preemptions["rows"]} == {
        (arm, dataset)
        for arm in ("native", "bidkv")
        for dataset in ("mmlu", "hle", "frontier", "swe", "terminal")
    }
    custody = {
        item["path"]: item["sha256"]
        for item in read(root / "five-results-v1/controlled-originals-index.json")[
            "files"
        ]
    }
    for row in preemptions["rows"]:
        assert row["before"] == row["after"] == row["delta"] == 0
        for position, checksum in zip(
            ("before", "after"), row["source_sha256"], strict=True
        ):
            name = row["arm"] + "-" + row["dataset"] + "/" + position + "-metrics.txt"
            assert custody[name] == checksum
    return {
        "historical_pressure_pair_valid": False,
        "native_successful_requests": 8,
        "candidate_tps": None,
        "full_solver_exact_preemption_audit_stages": 10,
        "controlled_metric_sha_bindings": 20,
        "controlled_metrics_publicly_replayed": False,
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).parent)))
