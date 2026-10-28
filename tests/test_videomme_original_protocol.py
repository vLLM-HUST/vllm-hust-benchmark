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
    assert not spec["source_checkbox_eligible"]  # A config alone is never evidence.
    evidence = json.loads(
        (ROOT / spec["source_completion_evidence"]).read_text("utf-8")
    )
    assert (
        spec["completed_native_calls"]
        == evidence["native_execution"]["executed_profile_tasks"]
        == 54
    )
    assert evidence["checked"] and evidence["native_scoring_completed"]
    assert spec["remaining"] == [] and spec["usage_boundary"]


def test_recorded_native_delivery_retains_sample_and_evidence_boundaries():
    evidence = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/videomme-original-readiness-20261028.json"
        ).read_text("utf-8")
    )
    native = evidence["native_execution"]
    assert evidence["original_rows_all11_fields_equal"] == 2700
    assert native["unique_original_questions"] == 27
    assert native["unique_original_videos"] == 9
    assert [profile["fixed_tasks"] for profile in native["profiles"]] == [27, 27]
    assert all(profile["actual_exit_code"] == 0 for profile in native["profiles"])
    assert evidence["independent_saved_native_audit"]["new_model_calls"] == 0
    assert not native["other_domain_quality_coverage_claim"]
    assert not native["question_replacement_or_budget_reduction"]
    assert (
        evidence["current_source_scoped_ci_and_maintained_consumer_reuse"][
            "only_changed_source_descriptor"
        ]
        == "BIGCODEBENCH"
    )


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


def test_acquisition_plan_does_not_download_or_create_paths(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            str(ROOT / "scripts/prepare_videomme_original_assets.py"),
            "--cache",
            str(tmp_path / "absent-cache"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY" and plan["original_files"] == 24
    assert plan["original_release_bytes"] == 101002235118
    assert not plan["native_media_ready"] and not plan["source_checkbox"]
    assert list(tmp_path.iterdir()) == []


def test_download_refuses_without_explicit_research_terms_and_byte_budget(tmp_path):
    entry = ROOT / "scripts/prepare_videomme_original_assets.py"
    base = [
        sys.executable,
        "-I",
        str(entry),
        "--cache",
        str(tmp_path / "absent-cache"),
        "--action",
        "download",
    ]
    result = subprocess.run(base, capture_output=True, text=True)
    assert result.returncode != 0 and "academic-only" in result.stderr
    result = subprocess.run(
        [*base, "--accept-academic-only"], capture_output=True, text=True
    )
    assert result.returncode != 0 and "byte budget" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_annotation_regular_view_does_not_imply_video_acquisition(tmp_path):
    import hashlib

    spec = importlib.util.spec_from_file_location(
        "videomme_assets", ROOT / "scripts/prepare_videomme_original_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    raw = b"software-fixture-only"
    (snapshot / "fixture.bin").write_bytes(raw)
    manifest = {
        "files": [
            {
                "path": "fixture.bin",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "role": "data",
            },
            {
                "path": "absent-video.zip",
                "bytes": 999,
                "sha256": "fixture-not-publisher",
                "role": "source_archive",
            },
        ]
    }
    target = tmp_path / "data-view"
    assert module.prepare_data_view(snapshot, target, manifest) == 1
    assert (snapshot / "fixture.bin").samefile(target / "fixture.bin")
    assert not (target / "absent-video.zip").exists()
    with pytest.raises(ValueError, match="missing"):
        module.verify_files(snapshot, manifest["files"])
