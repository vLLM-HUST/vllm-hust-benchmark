"""Reject diagnostic evidence being promoted to a control or model score."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "reports/terminal-cython-single-trace-20261027"


def load():
    spec = importlib.util.spec_from_file_location(
        "cython_trace_verify", DIRECTORY / "verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_trace():
    assert load().verify()["status"] == "passed"


@pytest.mark.parametrize(
    "key,value",
    [
        ("new_Agent_attempts", 1),
        ("new_model_calls", 1),
        ("new_control_scores", 1),
        ("task_reward", 0),
        ("original_trace_recovered", True),
        ("model_qualification_passed", True),
        ("original_environment_bytes_reproduced", True),
        ("test_or_oracle_modified", True),
        ("all_containers_released", False),
    ],
)
def test_trace_rejects_overstatement(key, value):
    data = json.loads((DIRECTORY / "summary.json").read_text())
    data[key] = value
    with pytest.raises(AssertionError):
        load().validate(data)
