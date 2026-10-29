import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "serving_cost", ROOT / "scripts/recompute_fixed018_serving_cost.py"
)
CALC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALC)
INPUT = ROOT / "reports/acceptance-gaps-20261029/recovered-serving-inputs.json"


@pytest.fixture
def inputs():
    return json.loads(INPUT.read_text())


def test_committed_projection_reproduces_summary(inputs):
    actual = CALC.calculate(inputs)
    actual["public_input_sha256"] = CALC.digest(INPUT.read_bytes())
    expected = json.loads(INPUT.with_name("recovered-serving-summary.json").read_text())
    assert actual == expected
    assert actual["improvements_percent"][
        "same_rate_compute_rental_cost"
    ] == pytest.approx(46.41779977)
    assert actual["improvements_percent"][
        "synchronized_window_output_throughput"
    ] == pytest.approx(86.62914096)
    assert actual["M06_pass"] is actual["utilization_improvement_pass"] is None
    assert actual["arms"]["B0"]["reserved_cards_verified"]
    assert actual["arms"]["B0"]["successful_output_tokens"] == 51200


def test_utilization_distinguishes_exact_alignment_and_conditional_bounds(inputs):
    b0 = CALC.time_weighted_utilization(inputs["B0"])
    b1 = CALC.time_weighted_utilization(inputs["B1"])
    assert b0["mean_percent"] is None
    bounds = b0["conditional_lifecycle_bounds"]
    assert bounds["mean_percent_lower"] == pytest.approx(9.42445480)
    assert bounds["mean_percent_upper"] == pytest.approx(16.71771426)
    assert b1["mean_percent"] == pytest.approx(22.39455412)
    assert bounds["mean_percent_upper"] < b1["mean_percent"]
    inputs["B0"]["controller_sha256"] = "unverified"
    assert CALC.lifecycle_utilization_bounds(inputs["B0"]) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "crossing",
        "missing",
        "duplicate",
        "nan",
        "tokens",
        "input",
        "identity",
        "allocation",
        "monitor",
    ],
)
def test_reject_incomplete_or_mismatched_evidence(inputs, mutation):
    arm = inputs["B0"]
    if mutation == "crossing":
        arm["requests"][0]["last_output_perf"] = arm["synchronized_end_perf"] + 1
    elif mutation == "missing":
        arm["requests"].pop()
    elif mutation == "duplicate":
        arm["requests"][0] = copy.deepcopy(arm["requests"][1])
    elif mutation == "nan":
        arm["requests"][0]["ttft_seconds"] = float("nan")
    elif mutation == "tokens":
        arm["requests"][0]["output_tokens"] = 255
    elif mutation == "input":
        arm["requests"][0]["input_tokens"] += 1
    elif mutation == "identity":
        arm["materialized_input_sha256"] = "another cohort"
    elif mutation == "allocation":
        arm["reserved_physical_devices"].append(7)
    else:
        arm["npu_samples"][1]["wall_time"] = arm["npu_samples"][0]["wall_time"]
    with pytest.raises(ValueError):
        CALC.calculate(inputs)


def test_projection_only_contains_scalar_requests(inputs):
    allowed = {
        "index",
        "start_perf",
        "ttft_seconds",
        "decode_seconds",
        "input_tokens",
        "output_tokens",
        "last_output_perf",
    }
    for name in ("B0", "B1"):
        arm = inputs[name]
        assert arm["verified_selected_manifest_members"] == 5
        assert arm["reserved_physical_devices"] == [3, 6]
        for row in arm["requests"]:
            assert set(row) == allowed
            assert all(isinstance(value, (int, float)) for value in row.values())


def test_time_weighting_is_not_sample_mean():
    arm = {
        "reserved_physical_devices": [3],
        "synchronized_start_perf": 0,
        "synchronized_end_perf": 10,
        "clock_brackets": [{"wall_ns": 0, "perf_before_ns": 0, "perf_after_ns": 0}],
        "npu_samples": [
            {"wall_time": 0, "ai_core_percent": {"3": 10}},
            {"wall_time": 9, "ai_core_percent": {"3": 90}},
            {"wall_time": 10, "ai_core_percent": {"3": 0}},
        ],
    }
    assert CALC.time_weighted_utilization(arm)["mean_percent"] == 18
