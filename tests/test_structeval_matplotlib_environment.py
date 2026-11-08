"""Software contracts only; no image build, model, original task or renderer."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_minimal_recipe_contains_no_task_or_other_benchmark_runtime():
    recipe = (
        ROOT / "configs/workload-native/structeval-matplotlib-cpu.Dockerfile"
    ).read_text()
    pins = (
        ROOT / "configs/workload-native/structeval-matplotlib-py311-observed.txt"
    ).read_text()
    assert len([line for line in pins.splitlines() if "==" in line]) == 12
    assert "matplotlib==3.11.2" in pins
    assert "USER 65534:65534" in recipe
    assert "pip check --python /opt/venv/bin/python" in recipe
    assert "COPY data" not in recipe and "COPY models" not in recipe
    assert "appworld==" not in recipe and "torch==" not in pins


def test_incomplete_closure_rejected_before_native_imports(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "matplotlib_environment",
        ROOT / "scripts/verify_structeval_matplotlib_environment.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lock = tmp_path / "incomplete.txt"
    lock.write_text("matplotlib==3.11.2\n")
    with pytest.raises(ValueError, match="complete observed12"):
        module.verify(lock)
