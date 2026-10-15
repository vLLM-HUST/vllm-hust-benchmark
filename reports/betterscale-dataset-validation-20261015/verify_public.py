"""Verify aggregate and per-case consistency without private question/answer payloads."""

import gzip
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    report = json.loads((root / "report.json").read_text())
    rows = json.loads(
        gzip.decompress((root / "per-case-outcomes.json.gz").read_bytes())
    )
    mmlu = rows["mmlu"]
    assert (
        len(mmlu)
        == len({r["case_id"] for r in mmlu})
        == report["mmlu"]["cases"]
        == 12032
    )
    assert sum(r["correct"] for r in mmlu) == report["mmlu"]["correct"] == 9611
    for field, predicate in (
        ("invalid_answers", lambda r: not r["valid"]),
        ("transport_failures", lambda r: not r["transport_success"]),
        ("length_limited", lambda r: r["length_limited"]),
    ):
        assert sum(predicate(r) for r in mmlu) == report["mmlu"][field]
    frontier = rows["frontier"]
    assert len(frontier) == len({r["case_id"] for r in frontier}) == 160
    for track, expected in report["frontier"]["tracks"].items():
        cases = [r for r in frontier if r["track"] == track]
        assert len(cases) == expected["total"]
        assert sum(r["correct"] is True for r in cases) == expected["successes"]
        assert (
            sum(r["judge_status"] == "invalid_judge" for r in cases)
            == expected["judge_invalid"]
        )
        for role in ("solver", "judge"):
            assert (
                sum(r[role + "_truncated"] for r in cases)
                == expected[role + "_truncated"]
            )
    scores = [
        r["score"]
        for r in frontier
        if r["track"] == "research" and r["score"] is not None
    ]
    assert len(scores) == 57
    expected = report["frontier"]["tracks"]["research"][
        "mean_research_score_valid_only"
    ]
    assert abs(sum(scores) / len(scores) - expected) < 1e-12
    for dataset, n, input_tokens in (("mmlu", 200, 282265), ("frontier", 160, 62571)):
        bench = report[dataset + "_bench"]
        assert bench["completed"] == n and bench["failed"] == 0
        assert bench["total_output_tokens"] == n * 256
        assert bench["total_input_tokens"] == input_tokens
        assert abs(bench["output_throughput"] - n * 256 / bench["duration"]) < 1e-9
        assert report[dataset + "_release"]["exit"] == 0
    assert all(json.loads((root / "verification.json").read_text()).values())
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    print(
        "PASS: 12032 MMLU outcomes, 160 Frontier outcomes, two load tests and release receipts."
    )
    print("Omitted raw response hashes do not substitute for independent regrading.")


if __name__ == "__main__":
    main()
