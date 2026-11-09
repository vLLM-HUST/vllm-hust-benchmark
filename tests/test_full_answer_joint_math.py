import copy
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("recompute_full_answer_joint")


def synthetic_data():
    # Synthetic arithmetic fixture, not a measurement or an acceptance result.
    arm = dict(
        role="B0",
        run="synthetic-b0",
        runtime=dict(vllm_version="0.18.0", ascend_version="0.18.0"),
        original_quality_plan_sha256="a" * 64,
        token_preflight_sha256="b" * 64,
        phase_seconds=100,
        cost=dict(
            price_basis=dict(
                basis="public_list_price_estimate_not_actual_invoice",
                accelerator_count=2,
                card_monthly_equivalent=18454,
                month_days_for_analysis=30,
            )
        ),
        rows=[
            dict(
                case_id=str(i),
                correct=i < 150,
                parsed=True,
                finish_reason="stop",
                output_tokens=101,
                ttft_ms=10,
                e2e_ms=1010,
                tpot_ms=10,
            )
            for i in range(200)
        ],
    )
    right = copy.deepcopy(arm)
    right.update(role="B1", run="synthetic-b1", phase_seconds=50)
    right["runtime"] = dict(vllm_version="0.28.1.dev", ascend_version="0.25.1rc2")
    right["rows"][150]["correct"] = True
    right["rows"][0]["correct"] = False
    right["rows"][1]["finish_reason"] = "length"
    return dict(arms=[arm, right])


@pytest.fixture
def data():
    return synthetic_data()


def test_joint_arithmetic_retains_quality_and_lengths(module, data):
    result = module.paired_metrics(data)
    assert result["paired_quality"] == dict(
        improved=1, regressed=1, difference_percentage_points=0
    )
    assert result["arms"][1]["truncated"] == 1
    assert result["arms"][1]["quality_by_finish"]["length"] == dict(total=1, correct=1)
    assert result["arms"][0]["successful_output_tokens"] == 20200
    assert result["output_throughput_change_percent"] == 100
    assert result["rental_cost_reduction_percent"] == 50
    assert result["arms"][0]["request_phase_cny"] == pytest.approx(
        36908 / (30 * 24 * 3600) * 100
    )
    assert result["arms"][0]["timing"]["ttft_ms"]["p99"] == 10


@pytest.mark.parametrize(
    "field,value",
    [("phase_seconds", 0), ("phase_seconds", float("nan"))],
)
def test_invalid_measurements_rejected(module, data, field, value):
    data["arms"][0][field] = value
    with pytest.raises(ValueError):
        module.paired_metrics(data)


def test_fixed_b0_and_same_price_and_inputs(module, data):
    wrong = copy.deepcopy(data)
    wrong["arms"][0]["runtime"]["vllm_version"] = "0.28.1"
    with pytest.raises(ValueError, match="B0 is fixed"):
        module.paired_metrics(wrong)
    wrong = copy.deepcopy(data)
    wrong["arms"][1]["cost"]["price_basis"]["month_days_for_analysis"] = 31
    with pytest.raises(ValueError, match="same price"):
        module.paired_metrics(wrong)
    wrong = copy.deepcopy(data)
    wrong["arms"][1]["token_preflight_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Input identity"):
        module.paired_metrics(wrong)


def test_no_missing_or_duplicate_tasks(module, data):
    wrong = copy.deepcopy(data)
    wrong["arms"][0]["rows"].pop()
    with pytest.raises(ValueError):
        module.paired_metrics(wrong)
    wrong = copy.deepcopy(data)
    for arm in wrong["arms"]:
        arm["rows"][1]["case_id"] = arm["rows"][0]["case_id"]
    with pytest.raises(ValueError, match="complete fixed200"):
        module.paired_metrics(wrong)


def test_metrics_not_borrowed_from_other_budget(module, data):
    data["arms"][0]["rows"][0]["output_tokens"] = 256
    with pytest.raises(ValueError, match="TPOT does not match"):
        module.paired_metrics(data)


def test_unparsed_never_correct(module, data):
    data["arms"][0]["rows"][0]["parsed"] = False
    with pytest.raises(ValueError, match="unparsed"):
        module.paired_metrics(data)
