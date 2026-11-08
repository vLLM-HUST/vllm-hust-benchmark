"""Software recipe fixtures only; no image build or AppWorld tasks."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_core_recipe_uses_verified_arm_bases_and_observed_not_author_lock():
    recipe = (ROOT / "configs/workload-native/appworld-core-cpu.Dockerfile").read_text()
    assert (
        "python@sha256:8f51ccb8569b75cf967ffe3e1638b9953042bf36142d7032cbb6f0696c9cd2c6"
        in recipe
    )
    assert (
        "uv@sha256:43b6b8ab14b48e7850828739e984661755f25cb48b5f3a812d846df8846ff5cf"
        in recipe
    )
    assert "USER 65534:65534" in recipe
    assert "NOT an author lock" in recipe
    copies = [line for line in recipe.splitlines() if line.startswith("COPY ")]
    assert len(copies) == 3
    assert all("data/" not in line and "model" not in line for line in copies)
    pins = [
        line
        for line in (
            ROOT / "configs/workload-native/appworld-core-client-py311-observed.txt"
        )
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    ]
    assert len(pins) == 83 and "appworld==0.1.3.post1" in pins


def test_verifier_rejects_unqualified_platform_before_imports(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "appworld_core_verify", ROOT / "scripts/verify_appworld_core_cpu_environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "sys", SimpleNamespace(version_info=(3, 10, 22)))
    with pytest.raises(ValueError, match="CPython3.11.17"):
        module.verify(tmp_path / "not-read")


def test_core_verifier_does_not_run_tasks_or_models():
    tree = ast.parse(
        (ROOT / "scripts/verify_appworld_core_cpu_environment.py").read_text()
    )
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(node.func, ast.Name)
        and node.func.id in {"AppWorld", "evaluate_task", "eval", "exec"}
        for node in calls
    )
