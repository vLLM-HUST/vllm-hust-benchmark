"""Verify a retained dependency-transport positive-control failure."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    whitelist = json.loads((HERE / "whitelist.json").read_text())
    for name, item in whitelist["files"].items():
        raw = (HERE / name).read_bytes()
        assert (
            len(raw) == item["bytes"]
            and hashlib.sha256(raw).hexdigest() == item["sha256"]
        )
    s = json.loads((HERE / "summary.json").read_text())
    assert s["status"] == "positive_control_failed_dependency_transport"
    assert s["task"] == "count-dataset-tokens"
    for value in s["controls"].values():
        assert value["reward"] == value["passed"] == 0
        assert value["failed"] == len(value["test_statuses"]) == 1 and value["released"]
    assert not s["cause"]["algorithm_or_ARM64_functional_failure_established"]
    assert s["cause"]["verifier_executed"]
    assert (
        s["cause"]["oracle_agent_exit_code"]
        == s["cause"]["oracle_wrapper_exit_code"]
        == 1
    )
    assert s["cause"]["oracle_harbor_exit_code"] == 0
    assert s["new_model_requests"] == s["new_Agent_attempts"] == 0
    assert (s["Terminal_model_attempts"], s["Terminal_unattempted"]) == (6, 83)
    assert (
        s["original_task_tests_oracle_versions_unchanged"] and not s["automatic_retry"]
    )
    assert s["original_budgets_seconds"] == dict(build=600, agent=900, verifier=900)
    assert not s["Agent_admission"]
    return dict(
        status="passed",
        positive_control="failed_dependency_transport",
        model_attempts=0,
    )


if __name__ == "__main__":
    print(json.dumps(verify()))
