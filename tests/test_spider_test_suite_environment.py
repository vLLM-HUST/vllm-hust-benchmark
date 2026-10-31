"""Dependency fixtures only; no original task, model, database or SQL execution."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location(
        "spider_environment", ROOT / "scripts/verify_spider_test_suite_environment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_control_does_not_claim_native_execution(monkeypatch):
    module = load()
    monkeypatch.setattr(module.metadata, "version", module.PINS.__getitem__)
    monkeypatch.setitem(sys.modules, "nltk", SimpleNamespace(word_tokenize=str.split))
    monkeypatch.setitem(
        sys.modules, "sqlparse", SimpleNamespace(parse=lambda _: ("fixture",))
    )
    monkeypatch.setattr(
        module.sqlite3,
        "connect",
        lambda *a, **k: pytest.fail("no SQL execution permitted"),
    )
    result = module.verify()
    assert result["original_tasks_executed"] == result["sql_statements_executed"] == 0
    assert result["model_calls"] == 0
    assert not result["source_checkbox_eligible"]


def test_environment_rejects_changed_dependency_pin(monkeypatch):
    module = load()
    monkeypatch.setattr(module.metadata, "version", lambda _: "unreviewed")
    monkeypatch.setitem(sys.modules, "nltk", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "sqlparse", SimpleNamespace())
    with pytest.raises(ValueError, match="pins differ"):
        module.verify()


def test_recipe_preserves_separate_native_execution_gate():
    recipe = (
        ROOT / "configs/workload-native/spider-test-suite-cpu.Dockerfile"
    ).read_text()
    assert "USER 65534:65534" in recipe
    assert "punkt_tab" in recipe
    assert "not an author-supplied lock" in recipe
    assert "COPY database" not in recipe
    assert "COPY evaluation.py" not in recipe
