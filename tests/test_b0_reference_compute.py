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
    result = MODULE.calculate(100, 10, 2, 300, 10, 4, 90)
    assert result["arms"][0]["eta_compute_b0_percent"] == 100
    assert result["arms"][1]["eta_compute_b0_percent"] == 150
    assert result["arms"][1]["equivalent_reference_seconds"] == 15


def test_boundary_and_failure_are_not_forced_to_pass():
    assert MODULE.calculate(100, 1, 1, 90, 1, 1, 90)["arms"][1]["reference_target_met"]
    assert not MODULE.calculate(100, 1, 1, 89, 1, 1, 90)["arms"][1][
        "reference_target_met"
    ]


@pytest.mark.parametrize("invalid", [0, -1, math.nan, math.inf, True, "1"])
def test_invalid_measurements_rejected(invalid):
    with pytest.raises(ValueError):
        MODULE.calculate(100, invalid, 2, 100, 1, 2, 90)


def test_recompute_preserves_historical_independent_reference_scope():
    result = MODULE.recompute()
    assert result["independent_outline_eta_ref_pass"] is None
    assert result["original_independent_reference_reclassified"] is False
    assert result["baseline_percent_is_normalization_identity"] is True
    assert result["new_model_calls_for_this_calculation"] == 0
    for arm in result["arms"]:
        assert arm["F"] == sum(result["component_totals"][arm["role"]].values())
        assert arm["C_TFLOPS_per_chip"] == arm["F"] / (
            arm["chips"] * arm["T_seconds"] * 1e12
        )
