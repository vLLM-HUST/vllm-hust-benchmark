import json
from pathlib import Path
from statistics import mean

import pytest


BASE = Path(__file__).resolve().parents[1] / "reports/vllm-hust-midterm-20261019"
EVIDENCE = BASE / "m01-recovery/request-assessment-20261022"


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def test_request_scope_and_resource_formula():
    result = read("m01-request-window-assessment.json")
    boundary = read("m01-request-task-boundaries.json")
    assert result["request_seconds"] == boundary["request_phase"]["duration_seconds"]
    assert 135 < result["request_seconds"] < 136
    for bound in ("lower", "upper"):
        terms = result["resource_mean_seconds"][bound]
        assert terms["communication"] == mean(
            row[f"{bound}_seconds"] for row in boundary["communication_reference_terms"]
        )
        assert result["empirical_attainment_percent"][bound] == pytest.approx(
            100 * max(terms.values()) / result["request_seconds"]
        )
    assert result["empirical_attainment_percent"]["upper"] < 50
    assert result["verdict"] == "below_internal_threshold"
    assert result["uninstrumented_attainment"] is None
    assert result["formal_b1_admission"] is False


def test_only_observer_tasks_outside_request():
    identities = read("m01-request-outside-task-identities.json")
    tasks = [task for rank in identities["outside_tasks"] for task in rank]
    assert len(tasks) == 7
    assert {t["task_type_name"] for t in tasks} == {
        "PROFILING_ENABLE",
        "PROFILING_DISABLE",
        "EVENT_RECORD",
    }
    boundary = read("m01-request-task-boundaries.json")
    for rank in boundary["ranks"]:
        assert rank["tables"]["TASK"]["nominal"]["crossing"] == 0
        comm = rank["tables"]["COMMUNICATION_OP"]["contracted_2ms"]
        assert comm["total"] == comm["inside"] == 83096


def test_hbm_coverage_and_report_value():
    result = read("m01-request-window-assessment.json")
    for device in result["hbm_expanded_window"]:
        assert len(device["channels"]) == 8
        assert all(c["coverage"] == 1 for c in device["channels"])
    text = (BASE / "中期测试报告.md").read_text()
    bounds = result["empirical_attainment_percent"]
    assert f"{bounds['lower']:.4f}%–{bounds['upper']:.4f}%" in text
