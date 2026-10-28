"""Software contracts only: no original data, media decoding or inference."""

import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return importlib.import_module("run_videomme_original")


@pytest.mark.parametrize("profile", ["Video-MME_8frame", "Video-MME_8frame_subs"])
@pytest.mark.parametrize("prepare_only", [False, True])
def test_offline_plan_never_reads_assets_or_calls_model(
    tmp_path, profile, prepare_only
):
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
            *(["--prepare-prompts-only"] if prepare_only else []),
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


def test_subtitle_overlay_is_explicit_and_version_checked(
    tmp_path, runner, monkeypatch
):
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    fake = SimpleNamespace(__file__=str(overlay / "pysubs2/__init__.py"))
    monkeypatch.setitem(sys.modules, "pysubs2", fake)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "1.8.0")
    monkeypatch.setattr(sys, "path", sys.path.copy())
    runner.prepare_subtitle_dependency(overlay)
    assert sys.path[0] == str(overlay.resolve())
    monkeypatch.setattr(
        runner.importlib.metadata, "version", lambda name: "0.0-fixture"
    )
    with pytest.raises(ValueError, match="never auto-upgrade"):
        runner.prepare_subtitle_dependency(overlay)
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: "1.8.0")
    fake.__file__ = str(tmp_path / "other-pysubs2/__init__.py")
    with pytest.raises(ValueError, match="declared isolated overlay"):
        runner.prepare_subtitle_dependency(overlay)


def test_original_prompt_preflight_keeps_frames_and_text_without_a_model(
    tmp_path, runner
):
    frame = tmp_path / "fixture-frame.bin"
    frame.write_bytes(b"software-fixture-not-real-media")
    prompt = [{"type": "image", "value": str(frame)} for _ in range(8)]
    prompt.append({"type": "text", "value": "unchanged software fixture"})
    dataset = SimpleNamespace(build_prompt=lambda index, video_llm: prompt)
    result = runner.prepare_original_prompts(dataset, [0, 1, 2], tmp_path)
    assert result["model_calls"] == 0 and not result["source_checkbox_eligible"]
    assert result["prepared_original_prompts"] == 3
    records = json.loads((tmp_path / "original-prompts-private.json").read_text())
    assert [record["source_index"] for record in records] == [0, 1, 2]
    assert all(record["original_prompt"] == prompt for record in records)
    assert all(len(record["frame_sha256"]) == 8 for record in records)


def test_prompt_preflight_rejects_incomplete_frames_without_a_completion_receipt(
    tmp_path, runner
):
    dataset = SimpleNamespace(build_prompt=lambda index, video_llm: [])
    with pytest.raises(ValueError, match="eight-frame"):
        runner.prepare_original_prompts(dataset, [0, 1, 2], tmp_path)
    assert list(tmp_path.iterdir()) == []
