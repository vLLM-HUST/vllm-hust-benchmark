"""Frozen metadata/protocol constraints only; no dataset acquisition or inference."""

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_original_profiles_keep_subtitle_and_no_subtitle_distinct():
    spec = json.loads(
        (ROOT / "configs/workload-native/videomme-original-protocol.json").read_text(
            "utf-8"
        )
    )
    assert [p["official_name"] for p in spec["profiles"]] == [
        "Video-MME_8frame",
        "Video-MME_8frame_subs",
    ]
    assert [p["use_subtitle"] for p in spec["profiles"]] == [False, True]
    assert all(p["frames"] == 8 and p["fps"] == -1 for p in spec["profiles"])
    assert not spec["source_checkbox_eligible"] and spec["completed_native_calls"] == 0
    assert spec["remaining"] and spec["usage_boundary"]


def test_frozen_selection_keeps_nine_videos_and_all_original_questions():
    spec = json.loads(
        (ROOT / "configs/workload-native/videomme-original-protocol.json").read_text(
            "utf-8"
        )
    )
    assert spec["original_tasks"] == 2700 and spec["original_videos"] == 900
    assert (
        len(spec["selected_original_positions"])
        == spec["selected_tasks_per_profile"]
        == 27
    )
    assert spec["selected_original_positions"] == [
        start + offset for start in (0, 900, 1800) for offset in range(9)
    ]
    assert all(len(ids) == 3 for ids in spec["selected_original_video_ids"].values())
    assert spec["planned_native_calls"] == 54


def test_original_table_plan_does_not_read_or_write_data(tmp_path):
    entry = ROOT / "scripts/prepare_videomme_original_view.py"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(entry),
            "--official-checkout",
            str(tmp_path / "absent-code"),
            "--snapshot",
            str(tmp_path / "absent-data"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY" and plan["model_calls"] == 0
    assert plan["original_tasks"] == 2700 and not plan["source_checkbox"]
    assert list(tmp_path.iterdir()) == []


def test_unverified_original_table_generator_cannot_execute(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "videomme_view", ROOT / "scripts/prepare_videomme_original_view.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(ValueError, match="Unverified original"):
        module.original_generator(b"raise RuntimeError('must never execute')")
    assert list(tmp_path.iterdir()) == []


def test_table_preparation_refuses_in_repository_data_before_optional_imports(
    tmp_path, monkeypatch
):
    spec = importlib.util.spec_from_file_location(
        "videomme_view_guard", ROOT / "scripts/prepare_videomme_original_view.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo = tmp_path / "repository"
    snapshot = repo / module.REVISION
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(module, "ROOT", repo)
    with pytest.raises(ValueError, match="External data root"):
        module.prepare_view(tmp_path / "absent-code", snapshot)
    assert list(snapshot.iterdir()) == []
