import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "long_context_audit", ROOT / "scripts/audit_fixed018_long_context.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def request(start=1, end=2, incoming=16384, success=True):
    return {
        "start": start,
        "end": end,
        "first_token": 1.5,
        "last_token": 1.8,
        "success": success,
        "prompt_tokens": incoming,
        "output_tokens": 2,
        "finish_reason": "length",
    }


def test_drain_and_failed_requests_do_not_become_closed_samples():
    result = AUDIT.summarize([request(), request(end=901), request(success=False)])
    assert result["input_ge16k_including_drain"] == 3
    assert result["input_ge16k_successful_closed_requests"] == 1
    assert result["failed_including_drain"] == 1
    assert result["fixed_4k_16k_pair_pass"] is None
    assert result["retrieval_correctness_pass"] is None


def test_length_is_actual_input_not_configured_model_capacity():
    result = AUDIT.summarize([request(incoming=4096), request(incoming=16383)])
    assert result["input_ge16k_including_drain"] == 0
    assert result["completion_time_diagnostic_windows"][0]["ttft_p99_seconds"] is None


def test_tpot_excludes_response_finalization_overhead():
    result = AUDIT.summarize([request(end=20)])
    assert result["completion_time_diagnostic_windows"][0][
        "tpot_p95_seconds"
    ] == pytest.approx(0.3)


def test_percentile_retains_empty_and_small_sample_semantics():
    assert AUDIT.percentile([], 0.99) is None
    assert AUDIT.percentile([1], 0.99) == 1
    assert AUDIT.percentile([0, 10], 0.95) == pytest.approx(9.5)


def test_all_five_original_windows_recompute():
    data = json.loads(
        (ROOT / "reports/long-context-fixed018-20261029/coverage.json").read_text()
    )
    assert data["new_model_runs"] == 0
    assert data["role"] == "B0"
    assert [a["summary"]["input_ge16k_including_drain"] for a in data["arms"]] == [
        22,
        52,
        91,
        127,
        168,
    ]
    for arm in data["arms"]:
        assert (
            arm["runtime"]["vllm_release"]
            == arm["runtime"]["vllm_ascend_release"]
            == "0.18.0"
        )
        assert arm["summary"] == AUDIT.summarize(arm["requests"])
        assert arm["summary"]["five_dataset_stability_pass"] is None
        assert arm["summary"]["failed_including_drain"] == 0
        assert len(arm["source_sha256"]) == 4
        for row in arm["requests"]:
            assert set(row) == {
                "start",
                "end",
                "first_token",
                "last_token",
                "success",
                "prompt_tokens",
                "finish_reason",
                "output_tokens",
            }
