"""Indentation-only local fixtures; not full official scoring or model tests."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "marble_compat", ROOT / "scripts/prepare_marble_evaluator_compat.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    raw = (
        b"class Fixture:\n    def parse(self, content):\n        try:\n            if content.startswith('{'):\n                return json.loads(content)\n"
        + entry.BLOCK
        + b"        else:\n            return {}\n"
    )
    monkeypatch.setattr(entry, "ORIGINAL_BYTES", len(raw))
    monkeypatch.setattr(entry, "ORIGINAL_SHA256", hashlib.sha256(raw).hexdigest())
    source = tmp_path / "source" / "evaluator.py"
    source.parent.mkdir()
    source.write_bytes(raw)
    return entry, source, raw


@pytest.mark.parametrize(
    "content,expected",
    [('{"rating":2}', {"rating": 2}), ("none", {}), ("{invalid}", {})],
)
def test_valid_missing_and_invalid_json_branches(fixture, content, expected):
    entry, source, raw = fixture
    patched, changed = entry.compatibility_copy(raw)
    assert len(changed) == 3
    module = ast.parse(patched)
    scope = {"json": json}
    exec(compile(module, "trusted-fixture-only", "exec"), scope)
    actor = scope["Fixture"]()
    actor.logger = SimpleNamespace(error=lambda _message: None)
    assert actor.parse(content) == expected
    assert source.read_bytes() == raw


def test_inert_plan_and_separate_disclosed_copy(fixture, tmp_path):
    entry, source, raw = fixture
    output = tmp_path / "compat"
    assert entry.prepare(source, output)["status"] == "PLAN_ONLY"
    assert not output.exists()
    report = entry.prepare(source, output, execute=True)
    assert not report["unmodified_official_source_claim"]
    assert not report["source_checkbox_eligible"]
    assert (output / "COMPATIBILITY.patch").is_file()
    assert source.read_bytes() == raw
    with pytest.raises(ValueError, match="fresh independent"):
        entry.prepare(source, output, execute=True)


def test_changed_source_is_not_blindly_patched(fixture):
    entry, _source, raw = fixture
    with pytest.raises(ValueError, match="identity"):
        entry.compatibility_copy(raw + b"changed")
