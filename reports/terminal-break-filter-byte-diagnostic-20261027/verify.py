"""Verify bounded filter-byte evidence, not original browser control success."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    whitelist = json.loads((HERE / "whitelist.json").read_text())
    for name, expected in whitelist["files"].items():
        assert Path(name).name == name
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == expected
    value = json.loads((HERE / "summary.json").read_text())
    assert value["process_exit"] == value["container_exit"] == 0
    assert value["original_oracle_filter_test_bytes_unchanged"] is True
    observation = value["filter"]
    assert observation["original_output_bytes"] == 41
    assert observation["filtered_bytes"] == 20
    assert observation["filter_exit_code"] == 0
    assert observation["browser_session_started"] is False
    assert observation["original_control_score_recomputed"] is False
    assert observation["original_alert_threshold_seconds_unchanged"] == 5
    boundary = value["boundaries"]
    for key in (
        "new_browser_sessions",
        "new_Agent_model_calls",
        "new_original_controls",
        "new_model_task_attempts",
        "original_positive_control_reward",
    ):
        assert boundary[key] == 0
    for key in (
        "original_five_second_threshold_unchanged",
        "original_r1_null_preserved",
        "fresh_transient_environment_not_claimed_original_snapshot",
        "diagnostic_pytest_not_original_control_success",
        "V098_original_snapshot_not_modified",
    ):
        assert boundary[key] is True
    assert value["diagnostic_pytest"] == {"tests": 1, "passed": 1, "failed": 0}
    assert value["released"] and value["sealed_files"] == 34
    return {
        "status": "passed",
        "diagnostic_only": True,
        "original_control_reward": 0,
        "new_model_attempts": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
