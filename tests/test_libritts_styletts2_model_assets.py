"""Small synthetic byte fixtures, never real model/data downloads."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/prepare_libritts_styletts2_model.py"
)
SPEC = importlib.util.spec_from_file_location("libritts_model_assets", SCRIPT)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


class Response(io.BytesIO):
    def __init__(self, raw, length=None):
        super().__init__(raw)
        self.headers = {} if length is None else {"Content-Length": str(length)}


def fixture_spec(raw=b"original fixture"):
    return {
        "path": "Models/LibriTTS/model.pth",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def test_complete_original_bytes_verified_and_existing_final_not_overwritten(tmp_path):
    spec = fixture_spec()
    digest = entry.download_one(
        tmp_path,
        spec,
        "https://example.invalid/original",
        opener=lambda *a, **k: Response(b"original fixture", spec["bytes"]),
    )
    assert (
        digest == spec["sha256"]
        and entry.verify(tmp_path / spec["path"], spec) == digest
    )
    with pytest.raises(ValueError, match="never overwritten"):
        entry.download_one(tmp_path, spec, "unused")


@pytest.mark.parametrize("raw", [b"short", b"original fixture plus excess"])
def test_short_or_excess_body_stops_and_retains_partial(tmp_path, raw):
    spec = fixture_spec()
    with pytest.raises(ValueError):
        entry.download_one(
            tmp_path,
            spec,
            "https://example.invalid/original",
            opener=lambda *a, **k: Response(raw),
        )
    assert not (tmp_path / spec["path"]).exists()
    assert (tmp_path / (spec["path"] + ".partial")).exists()


def test_declared_size_mismatch_creates_no_partial(tmp_path):
    spec = fixture_spec()
    with pytest.raises(ValueError, match="length differs"):
        entry.download_one(
            tmp_path,
            spec,
            "https://example.invalid/original",
            opener=lambda *a, **k: Response(b"wrong", 999),
        )
    assert not list(tmp_path.rglob("*.partial"))


def test_config_git_blob_and_escape_are_checked(tmp_path):
    raw = b"author config"
    spec = {
        "path": "config.yml",
        "bytes": len(raw),
        "git_blob": hashlib.sha1(
            b"blob " + str(len(raw)).encode() + b"\0" + raw
        ).hexdigest(),
    }
    assert (
        entry.download_one(
            tmp_path,
            spec,
            "https://example.invalid/config",
            opener=lambda *a, **k: Response(raw),
        )
        == hashlib.sha256(raw).hexdigest()
    )
    with pytest.raises(ValueError, match="contained"):
        entry.target_path(tmp_path, {"path": "../escape"})


def test_default_plan_never_loads_weights_or_calls_model():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY" and report["total_bytes"] == 771392385
    assert report["model_calls"] == 0 and not report["checkpoint_deserialized"]
    assert not report["source_checkbox_eligible"]
