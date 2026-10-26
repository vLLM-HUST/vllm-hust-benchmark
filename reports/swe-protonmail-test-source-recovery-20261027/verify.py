"""Verify public recovery aggregates offline; original scoring requires private evidence."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def validate(value):
    assert value["new_Agent_attempts"] == value["new_model_calls"] == 0
    assert value["test_source_restored"] is True
    assert value["tests_executed"] == value["parsed_tests"] == 0
    assert value["accepted_functional_score"] is None
    assert value["original_result"] == {"official": False, "parsed_tests": 0}
    assert value["recovery_result"]["official"] is False
    assert value["recovery_result"]["raw_tests_executed"] == 0
    for key in ("F2P", "P2P"):
        assert value["recovery_result"][key] == {
            "expected": 1,
            "passed": 0,
            "failed": 0,
            "missing": 1,
        }
    test = value["test_source"]
    assert test["bytes"] == 1033 and test["mode"] == "0644"
    assert (
        test["sha256"]
        == "e4de5749a0da6f4473831b7168ffa181f9796005dce57b382f3fdf51e55ab455"
    )
    assert test["before_and_after_equal"] and test["original_checkout_exit"] == 128
    patch = value["patch"]
    assert patch["bytes"] == 131723 and patch["apply_exit"] == 0
    assert (
        patch["sha256"]
        == "e7bf2eb1b6daac6b865dd7ed07bef399ded566172b931f5dd141369235764e87"
    )
    assert (
        patch["after_equals_graded"]
        and patch["all_three_affected_file_snapshots_equal_original"]
    )
    assert all(value["source_review"].values())
    assert value["failure"]["not_Agent_assertion_failure"] is True
    assert (
        value["released"]
        and value["controller_exited"]
        and value["original_files_unchanged"]
    )
    assert value["private_task_answer_patch_or_request_contents_included"] is False
    assert value["evidence"]["sealed_files"] == 32
    assert set(value["pretest_failures_preserved"]) == {"r1", "r2"}


def verify():
    for line in (HERE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = Path(name)
        assert not path.is_absolute() and ".." not in path.parts
        assert hashlib.sha256((HERE / path).read_bytes()).hexdigest() == digest
    value = json.loads((HERE / "summary.json").read_text())
    validate(value)
    independent = json.loads((HERE / "storage-public/summary.json").read_text())
    assert independent["original_result"] == value["original_result"]
    assert independent["recovery_result"] == value["recovery_result"]
    assert independent["Agent_attempts_added"] == value["new_Agent_attempts"]
    assert independent["model_requests_added"] == value["new_model_calls"]
    assert independent["test_source_sha256"] == value["test_source"]["sha256"]
    whitelist = json.loads((HERE / "storage-public/public-whitelist.json").read_text())
    assert len(whitelist["files"]) == 5
    for name, digest in whitelist["files"].items():
        path = Path(name)
        assert len(path.parts) == 1 and not path.is_absolute()
        assert (
            hashlib.sha256((HERE / "storage-public" / path).read_bytes()).hexdigest()
            == digest
        )
    index = json.loads((HERE / "restricted-original-sha-index.json").read_text())
    assert len(index) == value["evidence"]["sealed_files"]
    assert len({x["ordinal"] for x in index}) == len(index)
    assert all(len(x["sha256"]) == 64 for x in index)
    return {
        "status": "passed",
        "functional_score": None,
        "new_Agent_attempts": 0,
        "new_model_calls": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
