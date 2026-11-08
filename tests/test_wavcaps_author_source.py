"""Original software inventory and offline plan, not audio/model execution."""

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/wavcaps-author-source.json"


def test_complete_captioning_software_pins_exclude_data_and_weights():
    pins = json.loads(PINS.read_text("utf-8"))
    names = {item["path"] for item in pins["files"]}
    assert len(names) == len(pins["files"]) == 29
    assert sum(item["bytes"] for item in pins["files"]) == 191308
    assert {
        "captioning/test.py",
        "captioning/eval_metrics.py",
        "captioning/models/bart_captioning.py",
        "captioning/models/audio_encoder.py",
        "captioning/models/modeling_audio_encoder_decoder.py",
        "captioning/models/htsat.py",
        "captioning/settings/settings.yaml",
        "captioning/data_handling/caption_dataset.py",
    } <= names
    assert all(
        not name.startswith(("captioning/data/", "data/"))
        and Path(name).suffix in {".py", ".md", ".yaml"}
        for name in names
    )
    assert all(
        len(item["git_blob"]) == 40 and len(item["sha256"]) == 64
        for item in pins["files"]
    )
    assert not pins["native_execution_complete"]


def test_existing_source_preparer_supports_offline_wavcaps_plan(tmp_path):
    target = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_jsonschemabench_author_source.py"),
            "--source",
            "wavcaps",
            "--author-root",
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["source_id"] == "WAVCAPS"
    assert report["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert report["files"] == 29 and report["bytes"] == 191308
    assert report["model_calls"] == 0
    assert not report["source_checkbox_eligible"] and not target.exists()
