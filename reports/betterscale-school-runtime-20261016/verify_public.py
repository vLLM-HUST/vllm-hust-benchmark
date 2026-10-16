"""Check public aggregates and integrity, not independent answer regrading."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split()
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    report = json.loads((ROOT / "report.json").read_text())
    accuracy = report["accuracy"]
    assert accuracy["mmlu"]["cases"] == 12032
    assert accuracy["mmlu"]["correct"] == 9589
    assert accuracy["mmlu"]["transport_failures"] == 0
    for scope, n, correct, unresolved in (
        ("gold", 668, 173, 18),
        ("revision", 1143, 316, 28),
    ):
        summary = accuracy[scope]
        track = summary["tracks"][scope]
        assert (track["total"], track["successes"], track["unresolved"]) == (
            n,
            correct,
            unresolved,
        )
        assert track["accuracy"] == correct / n
        assert track["graded"] + unresolved == n
        assert track["solver_failures"] == track["judge_transport_failures"] == 0
        groups = [
            summary["groups"][f"{scope}/{modality}"] for modality in ("text", "image")
        ]
        for key in ("total", "successes", "unresolved"):
            assert sum(g[key] for g in groups) == track[key]
    for track, n, correct in (("olympiad", 100, 60), ("research", 60, 3)):
        result = accuracy["frontier"]["tracks"][track]
        assert result["total"] == n and result["successes"] == correct
        assert result["solver_failures"] == result["judge_transport_failures"] == 0
    for scope, n in (
        ("mmlu", 200),
        ("frontier", 160),
        ("gold-text", 200),
        ("revision-text", 200),
        ("gold-image", 93),
        ("revision-image", 118),
    ):
        perf = report["performance"][scope]
        assert perf["completed"] == perf["num_prompts"] == n and perf["failed"] == 0
        assert (
            abs(
                perf["total_output_tokens"] / perf["duration"]
                - perf["output_throughput"]
            )
            < 1e-6
        )
        if scope not in ("mmlu", "frontier"):
            assert perf["input_token_accounting_verified"]
    serialized = json.dumps(report)
    assert "/root/" not in serialized and "/Users/" not in serialized
    print(
        "School acceptance: all aggregates, denominators, six load tests and integrity checks passed"
    )


if __name__ == "__main__":
    verify()
