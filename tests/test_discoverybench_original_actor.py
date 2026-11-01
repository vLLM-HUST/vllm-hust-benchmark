"""Inert driver/input fixtures, not original Agent executions or model scoring."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "discovery_actor_driver", ROOT / "scripts/run_discoverybench_original_actor.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def metadata():
    return {
        "queries": [[{"qid": 0, "question": "unchanged fixture question"}]],
        "datasets": [{"name": "original.csv"}],
        "gold_hypo": "not an actor argument",
    }


def test_exact_question_zero_id_and_dataset_path_not_gold(tmp_path):
    (tmp_path / "original.csv").write_text("fixture only")
    query, paths = entry.task_inputs(metadata(), 0, tmp_path)
    assert query == "unchanged fixture question"
    assert paths == [str(tmp_path / "original.csv")]
    assert "not an actor argument" not in str((query, paths))


def test_duplicate_original_question_identity_rejected(tmp_path):
    value = metadata()
    value["queries"].append(value["queries"][0])
    with pytest.raises(ValueError, match="unique"):
        entry.task_inputs(value, 0, tmp_path)


def test_dataset_escape_rejected(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (tmp_path / "original.csv").write_text("fixture only")
    value = metadata()
    value["datasets"][0]["name"] = "../original.csv"
    with pytest.raises(ValueError, match="escaped"):
        entry.task_inputs(value, 0, root)


def test_modified_author_source_rejected_before_import(tmp_path):
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents/coder_utils.py").write_text("inert fixture, not imported")
    with pytest.raises(ValueError, match="unchanged original"):
        entry.verify_author(tmp_path)


def test_container_guard_precedes_assets_and_imports(monkeypatch):
    def reject():
        raise RuntimeError("fixture isolation rejection")

    monkeypatch.setattr(entry, "check_container", reject)
    with pytest.raises(RuntimeError, match="isolation rejection"):
        entry.execute(SimpleNamespace())


def test_default_is_plan_only(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fixture"])
    assert entry.main() == 0
    assert '"model_calls": 0' in capsys.readouterr().out
