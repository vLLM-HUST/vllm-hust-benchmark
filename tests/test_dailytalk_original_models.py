"""Artifact verification fixtures only, never neural inference evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "scripts/verify_dailytalk_original_models.py"


def module():
    spec = importlib.util.spec_from_file_location("dailytalk_model_files", ENTRY)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_plan_requires_no_files_libraries_devices_or_download(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            str(ENTRY),
            "--acoustic-checkpoint",
            str(tmp_path / "acoustic"),
            "--vocoder-checkpoint",
            str(tmp_path / "vocoder"),
            "--encoder-root",
            str(tmp_path / "encoder"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(r.stdout)
    assert (
        report["restore_step"] == 900000 and report["acoustic_variant"] == "history_guo"
    )
    assert (
        report["vocoder_variant"] == "universal"
        and report["encoder_embedding_dimension"] == 512
    )
    assert report["model_calls"] == 0 and not report["model_files_deserialized"]
    assert not report["source_checkbox_eligible"] and not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["sha256", "lfs_sha256", "git_blob"])
def test_streamed_identity_and_tamper_detection(tmp_path, kind):
    m = module()
    raw = b"offline model-file fixture, not a checkpoint"
    p = tmp_path / "fixture"
    p.write_bytes(raw)
    digest = (
        hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if kind == "git_blob"
        else hashlib.sha256(raw).hexdigest()
    )
    pin = {"bytes": len(raw), kind: digest}
    m.verify_file(p, pin)
    p.write_bytes(b"x" * len(raw))
    with pytest.raises(ValueError, match="digest differs"):
        m.verify_file(p, pin)


def test_official_model_roles_and_safe_encoder_scope():
    profile = json.loads(
        (
            ROOT / "configs/workload-native/dailytalk-original-model-assets.json"
        ).read_text()
    )
    assert profile["acoustic"]["bytes"] == 433765173
    assert profile["acoustic"]["history_window_turns"] == 10
    assert profile["vocoder"]["bytes"] == 55788858
    assert len(profile["encoder"]["files"]) == 13
    assert (
        profile["encoder"]["model"]
        == "sentence-transformers/distiluse-base-multilingual-cased-v1"
    )
    assert all(
        not x["path"].endswith((".bin", ".py")) for x in profile["encoder"]["files"]
    )
    assert {"model.safetensors", "2_Dense/model.safetensors"} <= {
        x["path"] for x in profile["encoder"]["files"]
    }
