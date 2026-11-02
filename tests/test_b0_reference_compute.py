"""Exercise the meaning and source binding of the new project reference."""

import importlib.util
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "reference_compute", ROOT / "scripts/recompute_b0_reference_compute.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_uses_work_and_all_chips_not_time_or_throughput_alone():
    result = MODULE.calculate(100, 10, 2, 300, 10, 4)
    assert result["arms"][0]["relative_compute_percent"] == 100
    assert result["arms"][1]["relative_compute_percent"] == 150


def test_regression_is_described_without_any_acceptance_verdict():
    result = MODULE.calculate(100, 1, 1, 90, 1, 1)
    assert result["arms"][1]["relative_compute_percent"] == 90
    assert result["C_gain_percent"] == -10
    for record in [result, *result["arms"]]:
        assert not any("target" in key or "margin" in key for key in record)


@pytest.mark.parametrize("invalid", [0, -1, math.nan, math.inf, True, "1"])
def test_invalid_measurements_rejected(invalid):
    with pytest.raises(ValueError):
        MODULE.calculate(100, invalid, 2, 100, 1, 2)


def test_recompute_preserves_historical_independent_reference_scope():
    result = MODULE.recompute()
    assert result["schema"] == "b0-effective-compute-comparison/v2"
    assert result["acceptance_scope"] == "descriptive_comparison_only"
    assert result["independent_outline_eta_ref_pass"] is None
    assert result["original_independent_reference_reclassified"] is False
    assert result["baseline_percent_is_normalization_identity"] is True
    assert result["new_model_calls_for_this_calculation"] == 0
    for arm in result["arms"]:
        assert arm["F"] == sum(result["component_totals"][arm["role"]].values())
        assert arm["C_TFLOPS_per_chip"] == arm["F"] / (
            arm["chips"] * arm["T_seconds"] * 1e12
        )
