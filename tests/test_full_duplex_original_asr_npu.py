"""No neural execution; test explicit device and checkpoint boundary handling."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/run_full_duplex_original_asr_npu.py"
)
SPEC = importlib.util.spec_from_file_location("duplex_npu_asr", SCRIPT)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


@pytest.mark.parametrize("version", ["v1.0", "v1.5"])
def test_plan_needs_no_model_files_or_neural_environment(tmp_path, version):
    command = [sys.executable, str(SCRIPT), "--version", version]
    for flag in ["omni-root", "reference-source", "tree", "nemo", "output"]:
        command.extend(["--" + flag, str(tmp_path / flag)])
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY_NO_MODEL_CALLS"
    assert plan["source_checkbox_eligible"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("devices", [[], ["cpu"], ["cuda:0"], ["npu:0", "cpu"]])
def test_neural_cpu_or_mixed_parameters_are_not_accepted(devices):
    model = argparse.Namespace(
        parameters=lambda: [argparse.Namespace(device=x) for x in devices]
    )
    with pytest.raises(ValueError, match="all neural parameters"):
        ENTRY.require_npu_parameters(model)


def test_npu_parameters_are_explicit():
    model = argparse.Namespace(parameters=lambda: [argparse.Namespace(device="npu:0")])
    assert ENTRY.require_npu_parameters(model) == ["npu:0"]


def test_wrong_checkpoint_fails_before_runtime_imports(tmp_path):
    path = tmp_path / "not-a-checkpoint"
    path.write_bytes(b"invented fixture")
    with pytest.raises(ValueError, match="checkpoint differs"):
        ENTRY.verify_checkpoint(path)
