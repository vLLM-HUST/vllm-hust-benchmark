"""Cross-check public controls against the independent sealed-evidence review."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    value = json.loads((HERE / "summary.json").read_text())
    review = json.loads((HERE / "independent-review.json").read_text())
    whitelist = json.loads((HERE / "storage-whitelist.json").read_text())
    raw = (HERE / "storage-summary.json").read_bytes()
    assert len(raw) == whitelist["allowed_files"]["summary.json"]["bytes"]
    assert (
        hashlib.sha256(raw).hexdigest()
        == whitelist["allowed_files"]["summary.json"]["sha256"]
    )
    storage = json.loads(raw)
    assert storage["original_seal_sha256"] == value["evidence_sha256"]["r2_seal"]
    assert storage["independent_review_sha256"] == whitelist["private_review_sha256"]
    assert storage["controls"]["oracle"]["reward"] == 0
    assert storage["Terminal_attempted_coverage_increment"] == 0
    assert value["evidence_sha256"]["r2_seal"] == review["seal_sha256"]
    assert value["r2"]["sealed_files"] == review["verified_files"] == 77
    assert (
        value["evidence_sha256"]["r2_build_plan"] == review["exact_build_plan_sha256"]
    )
    assert (
        hashlib.sha256((HERE / "derive_private.py.source").read_bytes()).hexdigest()
        == value["evidence_sha256"]["derivation_source"]
    )
    assert value["original_r1"]["reward"] is review["original_r1"]["reward"] is None
    assert (
        value["original_r1"]["nop_exit_code"]
        == review["original_r1"]["nop_exit"]
        == 130
    )
    assert value["original_r1"]["CTRF_present"] is False
    assert value["original_r1"]["oracle_started"] is False
    assert value["r2"]["new_image_build"] is review["r2"]["new_build"] is False
    assert (
        value["r2"]["reused_original_built_image"]
        is review["r2"]["build_reused"]
        is True
    )
    for role in ("nop", "oracle"):
        actual, checked = value["r2"]["controls"][role], review["r2"][role]
        assert actual["Harbor_exit_code"] == checked["Harbor_exit"] == 0
        assert actual["reward"] == checked["reward"] == 0
        assert actual["counts"]["tests"] == checked["test_count"] == 1
        assert actual["counts"]["passed"] == checked["passed"] == 0
        assert actual["counts"]["failed"] == checked["failed"] == 1
        assert len(actual["tests"]) == 1 and actual["tests"][0]["status"] == "failed"
        assert actual["owned_containers_released"] is True
    assert value["r2"]["controls"]["nop"]["control_status"] == "passed"
    assert value["r2"]["controls"]["oracle"]["control_status"] == "failed"
    assert (
        value["r2"]["oracle_wrapper_exit_code"]
        == review["r2"]["oracle"]["control_wrapper_exit"]
        == 1
    )
    assert value["original_budget"]["browser_alert_wait_seconds"] == 5
    assert value["task_oracle_tests_changed"] is False
    assert value["boundary"]["original_positive_control_passed"] is False
    assert value["boundary"]["model_task_score"] is None
    assert (
        value["boundary"]["new_model_calls"]
        == value["boundary"]["new_model_task_attempts"]
        == 0
    )
    assert value["boundary"]["total_model_tasks_attempted"] == 3
    assert value["boundary"]["unattempted_model_tasks"] == 86
    assert review["resources"]["owned_containers_released"] is True
    assert review["resources"]["controller_exited"] is True
    assert review["preflight_path_failure"]["failed_CPU_path_read_not_PASS"] is True
    assert value["private_task_oracle_answers_or_requests_included"] is False
    return {
        "status": "passed",
        "nop_reward": 0,
        "oracle_reward": 0,
        "positive_control_passed": False,
        "new_model_tasks": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
