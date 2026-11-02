"""Invented software/data fixtures only; no audio decoding, weights or inference."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "wavcaps_runner", ROOT / "scripts/run_wavcaps_original_captioning.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def test_offline_default_has_no_output_side_effect(tmp_path):
    output = tmp_path / "not-created"
    run = subprocess.run(
        [sys.executable, str(SPEC.origin), "--output-root", str(output)],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(run.stdout)
    assert report["status"] == "PLAN_ONLY_NO_MODEL_LOAD" and report["model_calls"] == 0
    assert not output.exists()


def test_bindings_change_only_two_declared_placeholders():
    raw = 'checkpoint_path = ""\naudio_path = ""\nanswer = generate(num_beams=3, max_length=30)\n'
    bound = ENTRY.bind_paths(raw, "/original/checkpoint", "/original/audio")
    assert bound.body[0].value.value == "/original/checkpoint"
    assert bound.body[1].value.value == "/original/audio"
    assert ast.dump(bound.body[2]) == ast.dump(ast.parse(raw).body[2])
    with pytest.raises(ValueError, match="placeholder differs"):
        ENTRY.bind_paths(
            raw.replace('audio_path = ""', 'audio_path = "modified"'), "x", "y"
        )
    with pytest.raises(ValueError, match="both original"):
        ENTRY.bind_paths('audio_path = ""', "x", "y")


def test_existing_execution_is_never_replayed(tmp_path):
    (tmp_path / "result-private.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="do not rerun"):
        ENTRY.execute(tmp_path)


def test_repository_output_rejected_before_input_or_model_access():
    with pytest.raises(ValueError, match="external server storage"):
        ENTRY.prepare({}, ROOT / "work/not-created-wavcaps")
    with pytest.raises(ValueError, match="external server storage"):
        ENTRY.execute(ROOT / "work/not-created-wavcaps")


@pytest.mark.parametrize("frozen", [True, False])
def test_prepare_binds_original_row_and_external_assets_without_models(
    tmp_path, monkeypatch, frozen
):
    checkout = tmp_path / "checkout"
    configs = checkout / "configs/workload-native"
    configs.mkdir(parents=True)
    monkeypatch.setattr(ENTRY, "ROOT", checkout)
    author = tmp_path / "author"
    (author / "captioning").mkdir(parents=True)
    raw = b'checkpoint_path = ""\naudio_path = ""\n'
    (author / "captioning/test.py").write_bytes(raw)
    row = {"id": "case1", "caption": "REFERENCE_NOT_A_MODEL_INPUT", "audio": "wav_path"}
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "meta.json").write_text(json.dumps({"data": [row]}), encoding="utf-8")
    checkpoint, encoder, audio = (
        tmp_path / name for name in ["checkpoint", "encoder", "audio"]
    )
    for path in [checkpoint, encoder, audio]:
        path.write_bytes(b"invented fixture, not weights or audio")
    bart = tmp_path / "bart"
    bart.mkdir()
    (bart / "config.json").write_bytes(b"invented model metadata")

    def identity(path, name=None):
        value = path.read_bytes()
        return {
            "path": name or path.name,
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
            "git_blob": hashlib.sha1(
                f"blob {len(value)}\0".encode() + value
            ).hexdigest(),
        }

    sources = tmp_path / "source-pins.json"
    sources.write_text(
        json.dumps(
            {"files": [identity(author / "captioning/test.py", "captioning/test.py")]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ENTRY, "SOURCES", sources)
    assets = tmp_path / "asset-pins.json"
    assets.write_text(
        json.dumps(
            {
                "checkpoint": identity(checkpoint),
                "audio_encoder": identity(encoder),
                "text_decoder": {
                    "revision": "invented-revision",
                    "files": [identity(bart / "config.json")],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ENTRY, "ASSETS", assets)
    profile = {
        "metadata": identity(metadata / "meta.json"),
        "files": [{"id": "case1", **identity(audio)}] if frozen else [],
    }
    (configs / "wavcaps-soundbible-original-fixed3-media.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    spec = {
        "author_root": str(author),
        "metadata_root": str(metadata),
        "checkpoint": str(checkpoint),
        "encoder_checkpoint": str(encoder),
        "bart_root": str(bart),
        "audio_file": str(audio),
        "component": "SoundBible",
        "case_id": "case1",
        "device": "npu:0",
        "memory_fraction": 0.1,
    }
    output = tmp_path / "prepared"
    prepared = ENTRY.prepare(spec, output)
    assert prepared["original_row"] == row and prepared["model_calls"] == 0
    assert prepared["frozen_media_identity_verified"] is frozen
    assert (output / "workspace/pretrained_models/audio_encoder/HTSAT.ckpt").samefile(
        encoder
    )
    assert (
        output
        / "hf-hub/models--facebook--bart-base/snapshots/invented-revision/config.json"
    ).samefile(bart / "config.json")
    assert not (output / "result-private.json").exists()
    with pytest.raises(FileExistsError):
        ENTRY.prepare(spec, output)
    assert (author / "captioning/test.py").read_bytes() == raw
    encoder_link = output / "workspace/pretrained_models/audio_encoder/HTSAT.ckpt"
    encoder_link.unlink()
    encoder_link.write_bytes(b"different fixture")
    with pytest.raises(ValueError, match="encoder link changed"):
        ENTRY.execute(output)
