"""Verify public agreement; optionally recompute from restricted original evidence."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    a = json.loads((root / "independent-review.json").read_text())
    b = json.loads((root / "storage-review.json").read_text())
    assert a["trajectory_sha256"] == b["trajectory_sha256"]
    assert a["source_seal_sha256"] == b["source_seal_sha256"]
    assert a["model_calls"] == b["model_responses"] == 100
    assert (
        a["tests"] == b["official_test_status_counts"] == {"PASSED": 117, "FAILED": 4}
    )
    assert a["resolved"] is b["official_resolved"] is False
    assert a["unresolved"] == b["official_grade_unresolved"] == 0
    assert a["finish_reasons"] == b["finish_reasons"] == {"stop": 100}
    if args.raw:
        trajectory = args.raw / "agent/task-000/trajectory-private.json"
        assert (
            hashlib.sha256(trajectory.read_bytes()).hexdigest()
            == a["trajectory_sha256"]
        )
        assert (
            hashlib.sha256((args.raw / "SHA256SUMS").read_bytes()).hexdigest()
            == a["source_seal_sha256"]
        )
        j = json.loads(trajectory.read_text())
        replies = [
            m["extra"]["response"] for m in j["messages"] if m["role"] == "assistant"
        ]
        assert len(replies) == 100
        assert dict(Counter(r["model"] for r in replies)) == a["actual_models"]
        assert (
            dict(Counter(r["choices"][0]["finish_reason"] for r in replies))
            == a["finish_reasons"]
        )
        tests = json.loads(
            next((args.raw / "grade").glob("*/workspace/output.json")).read_text()
        )["tests"]
        assert dict(Counter(t["status"] for t in tests)) == a["tests"]
        assert list(
            json.loads((args.raw / "grade/eval_results.json").read_text()).values()
        ) == [False]
    print(
        "PASS: public agreement"
        + (" and original-evidence recomputation" if args.raw else "")
    )


if __name__ == "__main__":
    main()
