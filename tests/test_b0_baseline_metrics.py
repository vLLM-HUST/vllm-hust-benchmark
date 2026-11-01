"""Formula invariants, including unequal outputs and deployment expenses."""

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METHOD = runpy.run_path(str(ROOT / "scripts/recompute_b0_baseline_metrics.py"))
calculate = METHOD["relative_metrics"]


def test_unequal_answers_separate_time_and_throughput_ratios():
    result = calculate(100, 10, 150, 5, 10, 5)
    assert result["batch_time_ratio"] == 2
    assert result["throughput_ratio"] == 3
    assert result["eta_b0_percent"] == 300
    assert result["throughput_gain_percent"] == 200
    assert not result["equal_output_tokens"]


def test_slower_result_is_not_clipped_or_confused_with_gain():
    result = calculate(100, 9, 100, 10, 9, 10)
    assert result["eta_b0_percent"] == pytest.approx(90)
    assert result["throughput_gain_percent"] == pytest.approx(-10)
    assert result["cost_reduction_percent"] < 0


def test_more_expensive_deployment_does_not_inherit_throughput_cost_saving():
    result = calculate(100, 10, 100, 5, 10, 15)
    assert result["throughput_ratio"] == 2
    assert result["cost_reduction_percent"] == pytest.approx(-50)


@pytest.mark.parametrize("invalid", [0, -1, float("nan"), float("inf"), True])
def test_invalid_measurements_are_rejected(invalid):
    with pytest.raises(ValueError):
        calculate(100, invalid, 100, 5, 10, 5)


def test_committed_example_matches_immutable_source():
    result = METHOD["recompute"](ROOT)
    assert result == json.loads((ROOT / METHOD["OUTPUT"]).read_text())
    assert result["metrics"]["eta_b0_percent"] == pytest.approx(190.4811043177129)
    assert (
        result["metrics"]["throughput_ratio"] != result["metrics"]["batch_time_ratio"]
    )
