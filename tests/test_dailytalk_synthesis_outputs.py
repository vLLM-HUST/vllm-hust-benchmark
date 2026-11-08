"""Saved-output audit fixtures do not load original models or synthesize speech."""

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from types import SimpleNamespace
import wave

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "dailytalk_outputs", ROOT / "scripts/verify_dailytalk_synthesis_outputs.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


@pytest.fixture
def saved(tmp_path, monkeypatch):
    identity = "0_0_d1"
    target = tmp_path / "results/900000/1"
    target.mkdir(parents=True)
    wav = target / f"{identity}.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setparams((1, 2, 22050, 0, "NONE", "not compressed"))
        stream.writeframes(b"\x01\x00" * 100)
    wav.with_suffix(".png").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", 2, 3)
    )
    report = {
        "status": "ORIGINAL_GUO_BATCH_SYNTHESIS_RETURNED_STRUCTURAL_OUTPUT_CHECKS_PASSED",
        "original_seed": 1234,
        "original_batch_size": 8,
        "original_history_window_turns": 10,
        "original_restore_step": 900000,
        "original_pitch_energy_duration_controls": [1.0, 1.0, 1.0],
        "weights_only_required": True,
        "unsafe_pickle_fallback_enabled": False,
        "native_model_forward_entries": {"acoustic": 1, "vocoder": 1},
        "outputs": [
            {
                "utterance_id": identity,
                "path": "results/900000/1/0_0_d1.wav",
                "original_plot_path": "results/900000/1/0_0_d1.png",
                "frames": 100,
                "sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
            }
        ],
    }
    receipt = tmp_path / "native-execution/state-private.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps(report), encoding="utf-8")
    for name in ("900000.pth.tar", "generator_universal.pth.tar"):
        (tmp_path / name).write_bytes(b"fixture, not original weights")
    monkeypatch.setattr(
        ENTRY,
        "load_inputs",
        lambda _: (
            [b"original fixture row"],
            {"utterances": [{"utterance_id": identity}]},
        ),
    )
    monkeypatch.setattr(ENTRY, "verify_file", lambda *_: None)
    return (
        SimpleNamespace(action="verify", stage=tmp_path, models_root=tmp_path),
        report,
        receipt,
    )


def test_default_plan_needs_no_files():
    result = ENTRY.audit(SimpleNamespace(action="plan"))
    assert result["audit_model_calls"] == 0
    assert not result["source_checkbox_eligible"]


def test_all_pcm_and_native_batch_coverage(saved):
    result = ENTRY.audit(saved[0])
    assert result["wav_files_full_pcm_decoded"] == 1
    assert result["wav_files_nonzero"] == 1
    assert result["minimum_frames"] == result["maximum_frames"] == 100
    assert result["audit_model_calls"] == 0
    assert not result["source_checkbox_eligible"]


@pytest.mark.parametrize(
    "change", ["history", "batches", "duplicate", "escape", "hash", "frames"]
)
def test_invalid_saved_native_evidence_rejected(saved, change):
    args, report, receipt = saved
    if change == "history":
        report["original_history_window_turns"] = 2
    elif change == "batches":
        report["native_model_forward_entries"]["vocoder"] = 0
    elif change == "duplicate":
        report["outputs"].append(dict(report["outputs"][0]))
    elif change == "escape":
        report["outputs"][0]["path"] = "../outside.wav"
    elif change == "hash":
        report["outputs"][0]["sha256"] = "0" * 64
    else:
        report["outputs"][0]["frames"] = 101
    receipt.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError):
        ENTRY.audit(args)


def test_invalid_plot_rejected(saved):
    args, _, _ = saved
    (args.stage / "results/900000/1/0_0_d1.png").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="PNG"):
        ENTRY.audit(args)


def test_extra_wav_rejected(saved):
    args, _, _ = saved
    (args.stage / "results/extra.wav").write_bytes(b"not an original output")
    with pytest.raises(ValueError, match="unexpected"):
        ENTRY.audit(args)


def test_actual_public_cli_receipt_matches_current_source():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/dailytalk-public-synthesis-audit-20261101.json"
        ).read_text("utf-8")
    )
    source = (ROOT / receipt["entry"]).read_text("utf-8").encode()
    assert hashlib.sha256(source).hexdigest() == receipt["entry_sha256"]
    assert receipt["exit_codes"] == [0, 0]
    assert not receipt["torch_imported"]
    assert receipt["old_native_receipt_unchanged"]
    result = receipt["result"]
    assert result["original_utterances"] == result["wav_files_full_pcm_decoded"] == 29
    assert result["audit_model_calls"] == 0
    assert not result["source_checkbox_eligible"]
