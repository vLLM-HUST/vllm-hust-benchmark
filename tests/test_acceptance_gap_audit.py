import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gap_audit", ROOT / "scripts/audit_acceptance_gaps.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def request(case_id=0, start=1, end=2, reason="stop", success=True):
    return {
        "case_id": case_id,
        "started": start,
        "ended": end,
        "finish_reason": reason,
        "success": success,
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def test_closed_cohort_excludes_crossing_and_failed():
    result = AUDIT.summarize_requests(
        [request(), request(1, end=11), request(2, success=False)], 0, 10
    )
    assert result["successful_in_interval"] == 1
    assert result["crossing_interval"] == result["failed_in_interval"] == 1
    assert result["output_tokens"] == 20


def test_length_stop_is_not_missing_and_does_not_pass_stability():
    result = AUDIT.summarize_requests([request(reason="length")], 0, 10)
    assert result["finish_reasons"] == {"length": 1}
    assert result["unknown_finish_reasons"] == 0
    assert result["five_dataset_stability_pass"] is None
    assert result["controlled_4k_16k_comparison_pass"] is None


def test_reject_duplicate_cohort():
    with pytest.raises(ValueError, match="Duplicate"):
        AUDIT.summarize_requests([request(), request()], 0, 10)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_cost(value):
    with pytest.raises(ValueError):
        AUDIT.unit_cost(value, 30, 2, 100, 1000)


def test_price_and_allocation_cancel_only_when_equal():
    assert AUDIT.unit_cost(720, 30, 2, 3600, 1_000_000) == 2
    assert AUDIT.unit_cost(720, 30, 4, 3600, 1_000_000) == 4


def test_gap_inventory_is_not_an_experiment_count():
    data = json.loads((ROOT / "reports/acceptance-gaps-20261029/gaps.json").read_text())
    assert data["open_acceptance_families"] == len(data["families"]) == 4
    assert data["required_new_experiment_count"] is None
    assert data["new_model_runs_started"] == 3
    assert data["new_model_runs_completed"] == 2
    assert data["new_service_start_attempts"] == 3
    assert data["startup_failures_before_requests"] == 1
    paired = json.loads(
        (
            ROOT / "reports/long-context-fixed018-paired-20261029/summary.json"
        ).read_text()
    )
    assert len(paired["arms"]) == data["new_model_runs_completed"]
    assert all(a["status"] == "retrieval_complete" for a in paired["arms"])
    assert paired["continuous_stability_pass"] is None
    assert set(data["covered_metrics"]) == {"M01", "M02", "M03", "M04", "M05", "M06"}


def test_conditional_cost_does_not_become_acceptance():
    data = json.loads(
        (
            ROOT / "reports/acceptance-gaps-20261029/pricing-and-conditional-cost.json"
        ).read_text()
    )
    for arm in ("B0", "B1"):
        row = data[arm]
        expected = AUDIT.unit_cost(
            18454, 30, 2, row["synchronized_seconds"], row["output_tokens"]
        )
        assert row["estimated_CNY_per_million_output_tokens"] == pytest.approx(expected)
    assert data["B0"]["reserved_cards_verified"] is False
    assert (
        data["cost_acceptance_pass"]
        is data["utilization_improvement_pass"]
        is data["M06_pass"]
        is None
    )


def test_quality_audit_keeps_actual_context_and_release_history():
    data = json.loads(
        (
            ROOT / "reports/acceptance-gaps-20261029/fixed018-quality-stability.json"
        ).read_text()
    )
    assert data["successful_in_interval"] == 200
    assert data["actual_context_max_tokens"] == 10515
    assert data["actual_context_at_least_16384"] == 0
    assert data["original_all_devices_released"] is False
    assert data["oom_marker_review"]["generic_platform_advisory_lines"] == [61, 511]
    assert data["oom_marker_review"]["non_advisory_oom_marker_lines"] == []
