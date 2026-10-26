"""Check public aggregates and preserve the grading-chain limitation."""

import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    review = json.loads((root / "independent-review.json").read_text())
    release = json.loads((root / "release-receipt.json").read_text())
    assert summary["total_dataset_tasks"] == 731
    assert summary["new_task_attempts"] == 1
    assert summary["recorded_official_resolved"] == 0
    assert summary["recorded_official_denominator"] == 1
    assert summary["official_unresolved_grading"] == 0
    assert summary["full_dataset_acceptance"] is False
    assert summary["vllm"] == summary["vllm_ascend"] == "0.18.0"
    assert summary["qualification"]["new_calls"] == 0
    assert summary["qualification"]["reused_calls"] == 5
    agent = summary["agent"]
    assert agent["status"] == "Submitted" and agent["agent_submitted"] is True
    assert agent["model_calls"] == 36 <= summary["budget"]["step_limit"]
    assert agent["elapsed_seconds"] < summary["budget"]["wall_seconds"]
    assert summary["budget"]["automatic_retries"] == 0
    assert summary["grading"]["official_result"] is False
    assert summary["grading"]["patch_application_attested"] is False
    assert summary["grading"]["model_calls"] == 0
    assert review["patch_chain"]["actual_application_exit_or_stderr_retained"] is False
    assert review["patch_chain"]["saved_patch_equals_official_input_snapshot"] is True
    assert review["official_grading"]["recorded_resolved"] is False
    assert review["patch_changed_by_review"] is False
    assert review["grading_reexecutions"] == 0
    for field in ("model_calls", "elapsed_seconds", "patch_bytes", "patch_sha256"):
        assert agent[field] == review["agent"][field]
    assert (
        agent["patch_sha256"]
        == summary["evidence_sha256"]["agent/task-000/patch-private.diff"]
    )
    for name, digest in summary["evidence_sha256"].items():
        if name in review["raw_evidence_sha256"]:
            assert digest == review["raw_evidence_sha256"][name]
    for field in ("sessions_drained", "all_devices_released", "seal_allowed"):
        assert release[field] is True
    assert release["status"] == "completed" and release["service_exit_code"] == 0
    assert summary["private_task_answers_patch_or_trajectory_included"] is False
    cached = json.loads((root / "cached-summary.json").read_text())
    audited = json.loads((root / "cached-independent-review.json").read_text())
    released = json.loads((root / "cached-release-receipt.json").read_text())
    assert audited["status"] == "passed_valid_negative_reusable"
    for field in ("model_calls", "elapsed_seconds", "patch_bytes", "patch_sha256"):
        assert cached["agent"][field] == audited["agent"][field]
    assert cached["agent"]["model_calls"] == 45
    grade = audited["official_grading"]
    assert grade["resolved"] is grade["independently_recomputed_official_rule"] is False
    assert grade["unresolved"] == [] and grade["model_calls"] == 0
    assert grade["run_exit_code"] == grade["grade_exit_code"] == 0
    assert grade["mocha_raw_counts"] == {"tests": 190, "passed": 189, "failed": 1}
    assert grade["official_parser_unique_counts"] == {"PASSED": 172, "FAILED": 1}
    assert grade["duplicate_names_collapsed"] == 17
    assert (
        cached["grading"]["parsed_test_counts"]
        == grade["official_parser_unique_counts"]
    )
    assert grade["fail_to_pass"] == {
        "expected": 1,
        "passed": 0,
        "failed": 1,
        "missing": 0,
    }
    assert grade["pass_to_pass"] == {
        "expected": 172,
        "passed": 172,
        "failed": 0,
        "missing": 0,
    }
    assert audited["initial_state"]["tracked_clean"] is True
    assert audited["initial_state"]["untracked_clean"] is True
    assert audited["initial_state"]["fixture_allowlist"] == []
    assert (
        audited["patch_and_scoring_chain"][
            "saved_official_snapshot_and_applied_input_identical"
        ]
        is True
    )
    assert cached["grading"]["official_result"] is False
    assert cached["full_dataset_acceptance"] is False
    assert audited["grading_reexecutions"] == audited["model_requests_by_review"] == 0
    for name, digest in audited["raw_evidence_sha256"].items():
        if name in cached["evidence_sha256"]:
            assert digest == cached["evidence_sha256"][name]
    assert released["result_review_sha256"] == audited["source_review_sha256"]
    assert released["status"] == "completed_released_seal_verified"
    assert released["sealed_files_verified"] == 12
    for field in ("sessions_drained", "all_devices_released", "seal_allowed"):
        assert released[field] is True
    return {
        "status": "passed",
        "official_result": False,
        "patch_application_attested": False,
        "summary_sha256": hashlib.sha256(
            (root / "summary.json").read_bytes()
        ).hexdigest(),
    }


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).parent)))
