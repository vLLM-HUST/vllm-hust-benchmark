"""Invented identity-only fixtures, not video decoding or model proof."""

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location(
        "ovo_media_verify", ROOT / "scripts/verify_ovobench_original_media.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.pop(0)


def test_identity_only_does_not_import_codec_or_claim_full_scope(tmp_path, monkeypatch):
    raw = b"invented file fixture, not valid media"
    (tmp_path / "0.mp4").write_bytes(raw)
    member = {
        "path": "0.mp4",
        "case_id": 0,
        "task": "ASI",
        "stage_index": None,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    selection = {
        "required_native_clips": [
            {**member, "path": f"{i}.mp4", "case_id": i} for i in range(103)
        ],
        "required_native_clip_count": 103,
    }
    monkeypatch.setitem(sys.modules, "av", None)
    result = entry.verify([member], selection, tmp_path)
    assert not result["files"][0]["full_video_decode_verified"]
    assert result["required_fixed_clip_count"] == 103
    assert not result["all_required_clips_registered"]
    assert not result["source_checkbox_eligible"] and result["model_calls"] == 0
    with pytest.raises(ValueError, match="fixed original selection"):
        entry.verify([{**member, "case_id": 1}], selection, tmp_path)
    (tmp_path / "0.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        entry.verify([member], selection, tmp_path)
