"""Software verifier fixtures only; no TeX compilation or native task."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location(
        "latex_environment", ROOT / "scripts/verify_structeval_latex_environment.py"
    )
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_verifier_preserves_software_only_scope(tmp_path, monkeypatch):
    loaded = module()
    binary = tmp_path / "inert"
    binary.write_bytes(b"inert software fixture, never executed")
    monkeypatch.setattr(loaded.metadata, "version", lambda _: "1.17.0")
    monkeypatch.setattr(loaded.shutil, "which", lambda _: str(binary))
    calls = []

    def lookup(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=str(binary) + "\n")

    monkeypatch.setattr(loaded.subprocess, "run", lookup)
    result = loaded.verify()
    assert len(calls) == 4
    assert all(call[0][0] == str(binary) and call[1]["check"] for call in calls)
    assert result["original_tasks_executed"] == result["model_calls"] == 0
    assert not result["source_checkbox_eligible"]
    assert not result["author_environment_lock_reproduced"]


def test_verifier_rejects_wrong_package_or_missing_compiler(monkeypatch):
    loaded = module()
    monkeypatch.setattr(loaded.metadata, "version", lambda _: "0.0")
    with pytest.raises(ValueError, match="version differs"):
        loaded.verify()
    monkeypatch.setattr(loaded.metadata, "version", lambda _: "1.17.0")
    monkeypatch.setattr(loaded.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="dependency"):
        loaded.verify()
