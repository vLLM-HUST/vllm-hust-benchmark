"""Original publisher preparation contracts, not speech synthesis evidence."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_source_plan_does_not_download_or_execute(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "dailytalk",
            "--author-root",
            str(tmp_path / "absent"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["files"] == 52 and report["bytes"] == 326159
    assert report["model_calls"] == 0 and not report["source_checkbox_eligible"]
    assert not list(tmp_path.iterdir())


def test_original_metadata_plan_preserves_audio_pending(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_dailytalk_original_metadata.py"),
            "--data-root",
            str(tmp_path / "absent"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["original_metadata_files"] == 11
    assert report["original_metadata_bytes"] == 12631639
    assert report["audio_files_acquired"] == 0
    assert not report["native_synthesis_or_scoring_executed"]
    assert not list(tmp_path.iterdir())


def test_original_metadata_budget_checked_before_download(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_dailytalk_original_metadata.py"),
            "--data-root",
            str(tmp_path / "absent"),
            "--action",
            "prepare",
            "--max-bytes",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "explicit metadata budget" in result.stderr
    assert not list(tmp_path.iterdir())


def test_publisher_files_exclude_weights_audio_and_transcript_repackaging():
    software = json.loads(
        (ROOT / "configs/workload-native/dailytalk-author-source.json").read_text()
    )
    metadata = json.loads(
        (
            ROOT
            / "workload-preparation/data/catalogs/dailytalk_original_metadata_files_v1.json"
        ).read_text()
    )
    assert software["revision"] == metadata["revision"]
    assert not {x["path"] for x in software["files"]} & {
        x["path"] for x in metadata["files"]
    }
    assert all(
        not x["path"].endswith((".zip", ".wav", ".pth", ".tar"))
        for x in software["files"] + metadata["files"]
    )
    names = {x["path"] for x in metadata["files"]}
    assert {
        "preprocessed_data/DailyTalk/val_phone.txt",
        "preprocessed_data/DailyTalk/val_frame.txt",
        "preprocessed_data/DailyTalk/speakers.json",
    } <= names
