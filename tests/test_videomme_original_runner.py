"""Software contracts only: no original data, media decoding or inference."""

import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("run_videomme_original")


@pytest.mark.parametrize("profile", ["Video-MME_8frame", "Video-MME_8frame_subs"])
def test_offline_plan_never_reads_assets_or_calls_model(tmp_path, profile):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_videomme_original.py"),
            "--official-checkout",
            str(tmp_path / "absent-code"),
            "--snapshot",
            str(tmp_path / "absent-data"),
            "--media-receipt",
            str(tmp_path / "absent-receipt"),
            "--frame-cache",
            str(tmp_path / "absent-frames"),
            "--output",
            str(tmp_path / "absent-output"),
            "--model",
            "fixture-not-model",
            "--profile",
            profile,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["model_calls"] == 0 and plan["frames_each"] == 8
    assert plan["selected_original_tasks"] == 27
    assert not plan["source_checkbox_eligible"]
    assert list(tmp_path.iterdir()) == []


def test_fixed_selection_retains_all_questions_and_durations(runner):
    records = [
        {"duration": duration, "video": f"{duration}-{video}"}
        for duration in ("short", "medium", "long")
        for video in range(300)
        for _ in range(3)
    ]
    expected = [start + offset for start in (0, 900, 1800) for offset in range(9)]
    assert runner.fixed_positions(records) == expected
    records[0]["duration"] = "long"
    with pytest.raises(ValueError, match="duration order"):
        runner.fixed_positions(records)


def test_scorer_missing_unresolved_or_replaced_tasks_are_not_complete(runner):
    records = [{"source_index": i, "score": i % 2} for i in range(3)]
    runner.validate_scores(records, [0, 1, 2])
    with pytest.raises(ValueError, match="every selected"):
        runner.validate_scores(records[:2], [0, 1, 2])
    records[1]["score"] = -1
    with pytest.raises(ValueError, match="Unresolved"):
        runner.validate_scores(records, [0, 1, 2])
    with pytest.raises(ValueError, match="identities"):
        runner.validate_scores(records, [0, 2, 3])


def test_cache_metadata_pin_does_not_rewrite_other_revision(tmp_path, runner):
    snapshot = tmp_path / "datasets--lmms-lab--Video-MME/snapshots" / runner.REVISION
    snapshot.mkdir(parents=True)
    assert runner.pin_native_cache(snapshot) == tmp_path
    ref = snapshot.parent.parent / "refs/main"
    assert ref.read_text() == runner.REVISION
    runner.pin_native_cache(snapshot)
    ref.write_text("another-existing-revision")
    with pytest.raises(ValueError, match="not overwritten"):
        runner.pin_native_cache(snapshot)
    assert ref.read_text() == "another-existing-revision"


def test_readiness_never_comes_from_annotation_only_receipt(tmp_path, runner):
    with pytest.raises(ValueError, match="Complete original-media"):
        runner.check_media_receipt(
            tmp_path, {"source_id": "VIDEOMME", "model_calls": 0}
        )
