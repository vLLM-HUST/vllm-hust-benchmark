"""Verify Circuit's original unscored trial and request accounting offline."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def verify():
    whitelist = json.loads((HERE / "whitelist.json").read_text())
    assert set(whitelist["files"]) == {
        "summary.json",
        "receipt-projection.json",
        "request-accounting.json",
        "restricted-evidence-index.json",
    }
    values = {}
    for name, expected in whitelist["files"].items():
        assert Path(name).name == name
        data = (HERE / name).read_bytes()
        assert len(data) == expected["bytes"]
        assert hashlib.sha256(data).hexdigest() == expected["sha256"]
        values[name] = json.loads(data)
    summary = values["summary.json"]
    accounting = values["request-accounting.json"]
    receipt = values["receipt-projection.json"]
    for key, value in accounting.items():
        assert summary[key] == value
    assert (
        summary["HTTP_requests"],
        summary["HTTP_success"],
        summary["HTTP_rejected"],
    ) == (11, 10, 1)
    assert (
        summary["HTTP_requests"] == summary["HTTP_success"] + summary["HTTP_rejected"]
    )
    assert (
        summary["trajectory_agent_steps"],
        summary["Harbor_n_episodes"],
        summary["completed_API_latency_entries"],
    ) == (9, 10, 9)
    assert (
        summary["HTTP_success"]
        == summary["trajectory_agent_steps"] + summary["output_length_fallback"]
    )
    for source_key, metric_key, delta_key in (
        ("total_prompt_tokens", "prompt_tokens", "prompt_tokens"),
        ("total_completion_tokens", "completion_tokens", "completion_tokens"),
    ):
        assert (
            accounting["API_metric_usage"][metric_key]
            - accounting["trajectory_usage"][source_key]
            == accounting["usage_delta"][delta_key]
        )
    assert accounting["usage_delta"] == {
        "prompt_tokens": 5449,
        "completion_tokens": 4096,
    }
    assert summary["context_tokens"] == 32768 < 28673 + summary["budget"]["max_tokens"]
    assert summary["budget"] == {
        "turns": 100,
        "max_tokens": 4096,
        "seed": 17,
        "agent_timeout_sec": 3600,
        "verifier_timeout_sec": 3600,
        "whole_trial_retries": 0,
        "litellm_sdk_retries": 0,
        "harbor_llm_layer_attempts": 3,
        "terminus_query_layer_attempts": 3,
    }
    assert summary["agent_seconds"] == 330.91977 < 3600
    for key in ("verifier", "reward", "CTRF"):
        assert summary[key] is None and receipt[key] is None
    assert (
        summary["exception_type"]
        == receipt["exception"]
        == "ContextLengthExceededError"
    )
    assert summary["official_scored"] is False
    boundary = summary["answer_boundary"]
    assert boundary["answer_write_command_found"] is False
    assert boundary["complete_answer_snapshot"] is False
    assert boundary["original_final_answer_file_SHA_unknown"] is True
    assert boundary["new_execution"] == 0
    assert (summary["Terminal_attempts"], summary["Terminal_unattempted"]) == (4, 85)
    assert (
        summary["prior3_unchanged"]
        and summary["new_Agent_model_verifier_controls"] == 0
    )
    assert receipt["sessions_drained"] and receipt["all_devices_released"]
    assert receipt["original_result_not_overwritten"]
    return {
        "status": "passed",
        "HTTP": 11,
        "trajectory_steps": 9,
        "reward": None,
        "coverage": "4/89",
    }


if __name__ == "__main__":
    print(json.dumps(verify()))
