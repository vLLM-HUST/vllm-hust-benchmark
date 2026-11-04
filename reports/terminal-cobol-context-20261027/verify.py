"""Verify original unscored trial, accounting and source-artifact boundaries."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = ("cobol-modernization", 22, 900, 5)


def verify():
    whitelist = json.loads((HERE / "whitelist.json").read_text())
    assert set(whitelist["files"]) == {
        "summary.json",
        "receipt-projection.json",
        "request-accounting.json",
        "restricted-evidence-index.json",
    }
    values = {}
    for name, item in whitelist["files"].items():
        raw = (HERE / name).read_bytes()
        assert (
            len(raw) == item["bytes"]
            and hashlib.sha256(raw).hexdigest() == item["sha256"]
        )
        values[name] = json.loads(raw)
    s, r, a = (
        values[n]
        for n in ("summary.json", "receipt-projection.json", "request-accounting.json")
    )
    for k, v in a.items():
        assert s[k] == v
    task, success, timeout, attempted = EXPECTED
    assert s["task"] == task
    assert (
        s["HTTP_success"]
        == s["trajectory_agent_steps"]
        == s["completed_API_latency_entries"]
        == success
    )
    assert s["HTTP_requests"] == s["Harbor_n_episodes"] == success + 1
    assert s["HTTP_rejected"] == 1 and s["output_length_fallback"] == 0
    assert s["usage_delta"] == {"prompt_tokens": 0, "completion_tokens": 0}
    assert (
        a["API_metric_usage"]["prompt_tokens"]
        == a["trajectory_usage"]["total_prompt_tokens"]
    )
    assert (
        a["API_metric_usage"]["completion_tokens"]
        == a["trajectory_usage"]["total_completion_tokens"]
    )
    assert s["context_tokens"] == 32768 < 28673 + s["budget"]["max_tokens"]
    assert s["budget"] == dict(
        turns=100,
        max_tokens=4096,
        seed=17,
        agent_timeout_sec=timeout,
        verifier_timeout_sec=timeout,
        whole_trial_retries=0,
        litellm_sdk_retries=0,
        harbor_llm_layer_attempts=3,
        terminus_query_layer_attempts=3,
    )
    assert s["agent_seconds"] < timeout
    for k in ("verifier", "reward", "CTRF"):
        assert s[k] is r[k] is None
    assert s["exception_type"] == r["exception"] == "ContextLengthExceededError"
    assert not s["official_scored"]
    assert (
        s["Terminal_attempts"] == attempted
        and s["Terminal_unattempted"] == 89 - attempted
    )
    assert s["new_Agent_model_verifier_controls"] == 0
    assert (
        r["sessions_drained"]
        and r["all_devices_released"]
        and r["original_result_not_overwritten"]
    )
    boundary = s["answer_boundary"]
    assert not boundary["complete_original_filesystem_snapshot"]
    if task == "cobol-modernization":
        assert boundary["derived_script_bytes"] == 6098
        assert boundary["original_file_SHA"] is boundary["original_file_mode"] is None
        assert not boundary["recovery_started"] and boundary["original_null_unchanged"]
    else:
        assert boundary["original_Agent_autonomously_attempted_OCR"]
        assert not boundary["operator_injected_OCR_or_answer"]
        assert not boundary["required_answer_artifact_written_in_recorded_commands"]
        assert boundary["actual_final_answer_file_SHA"] is None
        assert s["corrected_observation"]["actual_steps_field"] == 34
    return dict(status="passed", task=task, HTTP=s["HTTP_requests"], reward=None)


if __name__ == "__main__":
    print(json.dumps(verify()))
