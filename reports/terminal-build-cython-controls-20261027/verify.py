"""Offline integrity and semantic checks for the failed positive control."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def validate(summary, review):
    assert summary["task"] == "build-cython-ext"
    assert summary["architecture"] == "arm64"
    assert summary["source_revision"] == "7131e4375048a0e408a8fb404b5f499d726b695b"
    assert summary["build_exit_code"] == 0
    assert summary["model_calls"] == summary["new_model_task_attempts"] == 0
    assert summary["control_qualification_passed"] is False
    assert summary["original_tests_and_oracle_unchanged"] is True
    assert summary["original_amd64_image_equivalence_claimed"] is False
    assert summary["private_task_or_oracle_bodies_included"] is False
    assert summary["all_containers_absent_at_completion"] is True
    assert summary["budget"] == {
        "build_seconds": 600,
        "agent_seconds": 900,
        "verifier_seconds": 900,
    }
    assert summary["resources"] == {
        "cpus": 1,
        "memory_bytes": 2147483648,
        "cpuset": "24-31",
        "MTU": 1400,
    }
    original = summary["task_files_sha256"]
    derived = summary["derived_task_files_sha256"]
    assert set(original) == set(derived) and len(original) == 8
    assert {k for k in original if original[k] != derived[k]} == {
        "environment/Dockerfile",
        "task.toml",
    }
    names = []
    for role, passed in (("nop", 1), ("oracle", 10)):
        control = summary["controls"][role]
        tests = control["tests"]
        assert len(tests) == 11 and len({t["name"] for t in tests}) == 11
        assert all(t["status"] in ("passed", "failed") for t in tests)
        assert sum(t["status"] == "passed" for t in tests) == passed
        assert control["counts"] == {"passed": passed, "failed": 11 - passed}
        assert control["reward"] == 0 and control["process_exit_code"] == 0
        assert control["all_owned_containers_released"] is True
        names.append({t["name"] for t in tests})
    assert names[0] == names[1]
    failed = {
        t["name"]
        for t in summary["controls"]["oracle"]["tests"]
        if t["status"] == "failed"
    }
    assert failed == {"test_pyknotid_repository_tests"}
    assert summary["controls"]["nop"]["control_status"] == "passed"
    assert summary["controls"]["oracle"]["control_status"] == "failed"
    assert summary["pipeline_status"] == "failed_or_incomplete"
    restricted = summary["restricted_originals"]
    assert len(restricted["files"]) == restricted["files_verified"] == 66
    assert review["seal_files_verified"] == 66
    assert review["seal_sha256"] == restricted["seal_sha256"]
    assert review["build_image"] == summary["derived_image_id"]
    assert review["original_task_test_oracle_unchanged"] is True
    assert review["nested_observation"] == {
        "passed": 17,
        "failed": 1,
        "exit_code": 1,
        "full_internal_trace_preserved": False,
    }
    assert review["private_task_or_oracle_bodies_included"] is False


def main():
    rows = (HERE / "SHA256SUMS").read_text().splitlines()
    for row in rows:
        digest, name = row.split("  ", 1)
        path = HERE / name
        assert path.parent == HERE and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
    summary = json.loads((HERE / "summary.json").read_text())
    review = json.loads((HERE / "independent-review.json").read_text())
    validate(summary, review)
    assert (
        hashlib.sha256(
            (HERE / "derive_private_controls.py.source").read_bytes()
        ).hexdigest()
        == summary["derivation_source_sha256"]
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "files": len(rows),
                "control_qualified": False,
                "model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
