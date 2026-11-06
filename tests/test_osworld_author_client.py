"""Offline control fixtures, not desktop or model execution."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_osworld_author_client as module


def setup(tmp_path, monkeypatch):
    lock = tmp_path / "client.lock"
    lock.write_text("Fixture_Package==1.0\n")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"revision": "fixture-not-publisher"}))
    monkeypatch.setattr(module, "LOCK", lock)
    monkeypatch.setattr(module, "PROFILE", profile)
    calls = []
    monkeypatch.setattr(module, "verify_source", lambda *args: calls.append("verify"))
    monkeypatch.setattr(module, "check_imports", lambda root: calls.append("imports"))
    return calls


def test_plan_does_not_import_or_require_source(tmp_path, monkeypatch):
    calls = setup(tmp_path, monkeypatch)
    result = module.run(tmp_path / "absent", "plan")
    assert calls == []
    assert result["status"] == "PLAN_ONLY_NO_AUTHOR_IMPORTS"
    assert result["source_checkbox_eligible"] is False


def test_source_verification_precedes_imports(tmp_path, monkeypatch):
    calls = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "1.0")
    result = module.run(tmp_path, "check-imports")
    assert calls == ["verify", "imports"]
    assert result["native_tasks"] == 0
    assert result["desktop_env_instantiated"] is False


def test_version_drift_stops_before_author_imports(tmp_path, monkeypatch):
    calls = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "2.0")
    with pytest.raises(ValueError, match="version mismatch"):
        module.run(tmp_path, "check-imports")
    assert calls == ["verify"]


def test_source_only_action_never_checks_environment_or_imports(tmp_path, monkeypatch):
    calls = setup(tmp_path, monkeypatch)
    result = module.run(tmp_path, "verify-source")
    assert calls == ["verify"]
    assert result["status"] == "ORIGINAL_SOURCE_VERIFIED_NO_IMPORTS"


def test_bad_original_source_stops_before_imports(tmp_path, monkeypatch):
    calls = setup(tmp_path, monkeypatch)

    def reject(*args):
        raise ValueError("original source identity mismatch")

    monkeypatch.setattr(module, "verify_source", reject)
    with pytest.raises(ValueError, match="source identity mismatch"):
        module.run(tmp_path, "check-imports")
    assert calls == []


@pytest.mark.parametrize(
    "value",
    ["fixture @ file:///secret\n", "fixture>=1\n", "", "Some_Name==1\nsome-name==1\n"],
)
def test_lock_rejects_ambiguous_or_external_inputs(tmp_path, value):
    lock = tmp_path / "bad.lock"
    lock.write_text(value)
    with pytest.raises(ValueError):
        module.locked_packages(lock)


def test_original_exports_checked_without_instantiation(tmp_path, monkeypatch):
    imports = []

    def forbidden():
        raise AssertionError("must not instantiate any original class or grader")

    exports = SimpleNamespace(
        **{
            name: forbidden
            for name in [
                "DesktopEnv",
                "SetupController",
                "match_in_list",
                "is_cookie_deleted",
                "is_expected_tabs",
                "get_default_search_engine",
                "Qwen3VLAgent",
            ]
        }
    )

    def load(name):
        imports.append(name)
        return exports

    monkeypatch.setattr(module.importlib, "import_module", load)
    original = list(sys.path)
    module.check_imports(tmp_path)
    assert len(imports) == 5
    assert sys.path == original


def test_real_lock_preserves_qualified_author_versions():
    packages = module.locked_packages(module.LOCK)
    assert packages["pillow"] == "11.0.0"
    assert packages["pymupdf"] == "1.27.1"
    assert packages["pdfplumber"] == "0.11.9"
    assert packages["torch"] == "2.5.1"


def test_smoke_workflow_parses_and_registers_only_offline_client_test():
    path = module.ROOT / ".github/workflows/workload-data-smoke.yml"
    workflow = yaml.safe_load(path.read_text("utf-8"))
    events = workflow.get("on", workflow.get(True, {}))
    assert "workflow_dispatch" in events
    steps = workflow["jobs"]["cpu-correctness"]["steps"]
    matching = [
        step for step in steps if "test_osworld_author_client.py" in step.get("run", "")
    ]
    assert len(matching) == 1
    assert (
        matching[0]["run"] == "python -m pytest -q tests/test_osworld_author_client.py"
    )
