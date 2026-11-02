"""Invented ZIP controls only; real checkpoint remains on the server."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "wavcaps_checkpoint", ROOT / "scripts/verify_wavcaps_original_checkpoint.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def fixture_archive(tmp_path, member="fixture/data.txt"):
    path = tmp_path / "invented-checkpoint.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, "not weights or original benchmark material")
    raw = path.read_bytes()
    return path, {
        "checkpoint": {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "zip_members": 1,
        }
    }


def test_default_plan_no_model_import_or_download():
    result = subprocess.run(
        [sys.executable, str(SPEC.origin)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY_NO_MODEL_LOAD"
    assert report["checkpoint"]["bytes"] == 2028254247
    assert report["model_calls"] == 0
    assert not report["model_files_deserialized"]
    assert not report["source_checkbox_eligible"]


def test_same_byte_identity_and_zip_control(tmp_path):
    path, profile = fixture_archive(tmp_path)
    assert ENTRY.verify(path, profile) == 1
    profile["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest differs"):
        ENTRY.verify(path, profile)


def test_unsafe_member_rejected_without_extraction(tmp_path):
    path, profile = fixture_archive(tmp_path, "../outside.txt")
    with pytest.raises(ValueError, match="unsafe original checkpoint"):
        ENTRY.verify(path, profile)
    assert not (tmp_path.parent / "outside.txt").exists()


def test_wrong_member_count_rejected(tmp_path):
    path, profile = fixture_archive(tmp_path)
    profile["checkpoint"]["zip_members"] = 2
    with pytest.raises(ValueError, match="ZIP structure differs"):
        ENTRY.verify(path, profile)
