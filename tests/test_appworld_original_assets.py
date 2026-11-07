"""Offline fixtures, not native AppWorld task execution."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "appworld_assets", ROOT / "scripts/prepare_appworld_original_assets.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_plan_never_creates_files_or_downloads(tmp_path):
    output = tmp_path / "absent"
    result = MODULE.prepare(output, tmp_path / "client", tmp_path / "cache")
    assert not output.exists() and result["status"] == "PLAN_ONLY"
    assert result["command"][1:3] == ["download", "data"]
    assert result["source_checkbox_eligible"] is False


def test_existing_data_not_replaced(tmp_path, monkeypatch):
    monkeypatch.setattr(MODULE.sys, "platform", "linux")
    with pytest.raises(ValueError, match="never replace"):
        MODULE.prepare(tmp_path, tmp_path / "client", tmp_path / "cache", True)


def test_all_original_split_links_preserved(tmp_path):
    ids = {split: [str(index)] for index, split in enumerate(MODULE.SPLITS)}
    for values in ids.values():
        folder = tmp_path / "data/tasks" / values[0]
        folder.mkdir(parents=True)
        (folder / "original.json").write_text("{}")
    assert MODULE.audit_relations(tmp_path, ids.__getitem__)["unique_tasks"] == 4


@pytest.mark.parametrize("ids", [[], ["same", "same"], ["../escape"]])
def test_bad_relations_fail_without_replacement(tmp_path, ids):
    with pytest.raises(ValueError):
        MODULE.audit_relations(tmp_path, lambda split: ids)
