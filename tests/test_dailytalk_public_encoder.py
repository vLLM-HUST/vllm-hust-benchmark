"""Offline fabricated fixtures, never original-data or NPU execution evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "dailytalk_public_encoder", ROOT / "scripts/run_dailytalk_guo_encoder_npu.py"
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def inputs(tmp_path, monkeypatch):
    metadata = tmp_path / "original"
    directory = metadata / "preprocessed_data/DailyTalk"
    directory.mkdir(parents=True)
    raw = b"0_0_d1|0|{HH}|hello|neutral\n1_1_d1|1|{W}|world|neutral\n"
    (directory / "val_phone.txt").write_bytes(raw)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    (prepared / "val_phone.fixed3.private.txt").write_bytes(raw)
    rows = entry.select_rows(raw, {"0_0_d1", "1_1_d1"})
    contract = {
        "history_window_turns": 10,
        "feature_dimension": 512,
        "utterances": entry.history_plan(rows),
    }
    (prepared / "guo-input-contract.private.json").write_text(json.dumps(contract))
    monkeypatch.setattr(entry, "verify_metadata", lambda args: None)
    args = SimpleNamespace(
        prepared=prepared,
        metadata_root=metadata,
        split="val",
        source_file=None,
        output=tmp_path / "receipt",
    )
    return args, rows, contract


def test_default_plan_reads_no_inputs_or_models(tmp_path):
    command = [sys.executable, str(ROOT / "scripts/run_dailytalk_guo_encoder_npu.py")]
    for name in ["prepared", "metadata-root", "encoder-root", "output"]:
        command += ["--" + name, str(tmp_path / name)]
    result = json.loads(subprocess.check_output(command))
    assert result["status"] == "PLAN_ONLY_NO_MODEL_LOAD"
    assert result["model_calls"] == 0 and not any(tmp_path.iterdir())


def test_resource_window_and_device_checked_before_reading_inputs():
    with pytest.raises(ValueError, match="idle resource window"):
        entry.run(
            SimpleNamespace(
                action="execute", device="npu:0", resource_window_confirmed=False
            )
        )
    with pytest.raises(ValueError, match="NPU device"):
        entry.run(SimpleNamespace(action="verify", device="cpu"))


@pytest.mark.parametrize(
    "path", ["../escape.npy", "/escape.npy", r"text_emb\escape.npy", "other/escape.npy"]
)
def test_feature_paths_cannot_escape_text_emb(tmp_path, path):
    with pytest.raises(ValueError):
        entry.feature_path(tmp_path, path)


def test_original_bytes_order_and_history_are_bound(tmp_path, monkeypatch):
    args, rows, contract = inputs(tmp_path, monkeypatch)
    observed, _ = entry.load_inputs(args)
    assert observed == rows
    assert contract["utterances"][1]["history_ids"] == ["0_0_d1"]
    (args.prepared / "val_phone.fixed3.private.txt").write_bytes(b"changed task")
    with pytest.raises(ValueError, match="original bytes and order"):
        entry.load_inputs(args)


def test_saved_features_are_verified_without_a_model_rerun(tmp_path, monkeypatch):
    args, rows, contract = inputs(tmp_path, monkeypatch)
    args.output.mkdir()
    profile = json.loads(entry.PROFILE.read_text(encoding="utf-8"))
    receipt = {
        "status": "ORIGINAL_GUO_TEXT_FEATURES_GENERATED_ON_NPU_NOT_TTS",
        "encoder_revision": profile["encoder"]["revision"],
        "model_calls": 2,
        "feature_dimension": 512,
        "max_seq_length": 128,
        "model_dtype": "float32",
        "features": [],
    }
    for item in contract["utterances"]:
        path = entry.feature_path(args.prepared, item["feature_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.zeros(512, dtype=np.float32), allow_pickle=False)
        receipt["features"].append(
            {
                "utterance_id": item["utterance_id"],
                "path": item["feature_path"],
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (args.output / "execution-private.json").write_text(json.dumps(receipt))
    report = entry.verify_saved(args, rows, contract)
    assert report["feature_files"] == 2 and report["new_model_calls"] == 0
    assert not report["source_checkbox_eligible"]
    path.write_bytes(b"corrupted feature")
    with pytest.raises(ValueError):
        entry.verify_saved(args, rows, contract)
    path.write_bytes(b"x" * 65537)
    with pytest.raises(ValueError, match="bounded size"):
        entry.verify_saved(args, rows, contract)
    with path.open("wb") as stream:
        np.lib.format.write_array_header_1_0(
            stream, {"descr": "<f4", "fortran_order": False, "shape": (2**44,)}
        )
    receipt["features"][-1].update(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    (args.output / "execution-private.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        entry.verify_saved(args, rows, contract)


def test_real_public_verification_receipt_is_bound_and_does_not_claim_execution():
    receipt = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/dailytalk-public-encoder-entry-20261101.json"
        ).read_text(encoding="utf-8")
    )
    code = (ROOT / receipt["entry"]).read_text(encoding="utf-8")
    assert hashlib.sha256(code.encode()).hexdigest() == receipt["entry_sha256"]
    assert (
        receipt["saved_features_verified"]
        == receipt["original_rows_and_history_reconstructed"]
        == 29
    )
    assert receipt["exit_codes"] == [0, 0] and receipt["old_encoder_receipt_unchanged"]
    assert (
        receipt["new_model_calls"] == 0
        and not receipt["torch_imported_in_public_cli_controls"]
    )
    assert (
        not receipt["public_execute_path_reexecuted"]
        and not receipt["source_checkbox_eligible"]
    )
