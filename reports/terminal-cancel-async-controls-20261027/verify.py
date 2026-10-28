"""Verify only public control aggregates and custody bindings, without executing tests."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    raw = (HERE / "summary.json").read_bytes()
    summary = json.loads(raw)
    review = json.loads((HERE / "independent-review.json").read_text())
    index = json.loads((HERE / "restricted-evidence-index.json").read_text())
    assert hashlib.sha256(raw).hexdigest() == review["public_summary_sha256"]
    assert index["seal_sha256"] == review["seal_sha256"]
    assert index["files_verified"] == review["sealed_files_verified"] == 68
    assert index["original_failed_run_preserved"]
    assert not index["private_task_or_solution_content_included"]
    assert summary["task"] == review["task"] == "cancel-async-tasks"
    assert summary["task_revision"] == review["task_revision"] == index["task_revision"]
    assert summary["platform"] == review["platform"] == "linux/arm64"
    assert summary["harbor_version"] == review["harbor_version"] == "0.24.0"
    assert summary["model_calls"] == summary["model_task_attempts"] == 0
    assert summary["official_89_task_model_coverage"] == 0
    assert review["model_calls"] == review["model_task_attempts"] == 0
    assert (
        summary["full_dataset_acceptance"] is review["full_dataset_acceptance"] is False
    )
    assert summary["amd64_image_equivalence_claimed"] is False
    assert summary["private_task_or_solution_content_included"] is False
    assert summary["original_task_tests_solution_instructions_unchanged"]
    assert review["original_timing_thresholds_unchanged"]
    assert review["distinct_serial_containers"]
    assert review["readonly_transport_and_writable_verifier_logs_verified"]
    assert summary["changed_files"] == ["task.toml", "environment/Dockerfile"]
    for key, value in (
        ("cpu_limit", 1),
        ("memory_bytes", 2147483648),
        ("cpuset", "24-31"),
        ("network_mtu", 1400),
    ):
        assert summary[key] == review[key] == value
    assert len(summary["controls"]) == len(review["controls"]) == 2
    for result, checked, role, reward in zip(
        summary["controls"], review["controls"], ("nop", "oracle"), (0, 1), strict=True
    ):
        assert result["role"] == checked["role"] == role
        assert result["reward"] == checked["reward"] == reward
        assert result["tests"] == checked["tests"] == 6
        assert result["passed"] == checked["passed"] == 6 * reward
        assert result["failed"] == checked["failed"] == 6 * (1 - reward)
        assert result["skipped"] == checked["missing"] == 0
        assert (
            result["process_exit_code"]
            == result["model_calls"]
            == checked["model_calls"]
            == 0
        )
        assert result["all_owned_containers_released"] and checked["released"]
    assert review["no_reruns"]
    for name in ("reward", "agent_execution", "verifier"):
        assert summary["original_precontainer_failure"][name] is None
        assert review["original_startup_failure"][name] is None
    assert summary["original_precontainer_failure"]["containers_started"] == 0
    assert review["original_startup_failure"]["retained"]
    return {"status": "passed", "control_trials": 2, "model_task_attempts": 0}


if __name__ == "__main__":
    print(json.dumps(verify()))
