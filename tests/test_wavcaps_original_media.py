"""Invented container/header fixtures; actual FLAC proof is server-only."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "wavcaps_media", ROOT / "scripts/verify_wavcaps_original_media.py"
)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def fixture(tmp_path):
    metadata_root = tmp_path / "metadata"
    media_root = tmp_path / "media"
    metadata_root.mkdir()
    media_root.mkdir()
    rows, files = [], []
    for identity in ("one", "two", "three"):
        rate, samples = 32000, 32000
        packed = (rate << 44) | (15 << 36) | samples
        raw = (
            b"fLaC\x80\x00\x00\x22"
            + bytes(10)
            + packed.to_bytes(8, "big")
            + bytes(16)
            + b"fixture-not-real-audio"
        )
        path = media_root / (identity + ".flac")
        path.write_bytes(raw)
        files.append(
            {
                "id": identity,
                "path": path.name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "crc32": zlib.crc32(raw),
                "sample_rate": rate,
                "channels": 1,
                "bits_per_sample": 16,
                "total_samples": samples,
            }
        )
        rows.append(
            {
                "id": identity,
                "duration": 1.0,
                "audio": "wav_path",
                "caption": "fixture reference",
            }
        )
    metadata = metadata_root / "original.json"
    metadata.write_text(json.dumps({"num_captions_per_audio": 1, "data": rows}))
    profile = {
        "metadata": {
            "path": metadata.name,
            "bytes": metadata.stat().st_size,
            "sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        },
        "files": files,
    }
    return profile, metadata_root, media_root


def test_header_only_is_explicitly_not_full_audio_or_inference(tmp_path):
    profile, metadata, media = fixture(tmp_path)
    before = (metadata / "original.json").read_bytes()
    result = entry.verify(profile, metadata_root=metadata, media_root=media)
    assert len(result["files"]) == 3
    assert result["source_checkbox_eligible"] is False and result["model_calls"] == 0
    assert result["full_decode_requested"] is False
    assert all(not item["full_decode_pcm_md5_verified"] for item in result["files"])
    assert (metadata / "original.json").read_bytes() == before


def test_changed_member_is_not_accepted(tmp_path):
    profile, metadata, media = fixture(tmp_path)
    (media / "one.flac").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        entry.verify(profile, metadata_root=metadata, media_root=media)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("crc32", 0, "CRC"),
        ("total_samples", 1, "dimensions"),
        ("path", "../escape.flac", "escaping root"),
    ],
)
def test_bad_original_binding_fails(tmp_path, field, value, message):
    profile, metadata, media = fixture(tmp_path)
    profile["files"][0][field] = value
    with pytest.raises(ValueError, match=message):
        entry.verify(profile, metadata_root=metadata, media_root=media)


def test_fixed_original_order_not_replaced(tmp_path):
    profile, metadata, media = fixture(tmp_path)
    profile["files"].reverse()
    with pytest.raises(ValueError, match="first3 identities"):
        entry.verify(profile, metadata_root=metadata, media_root=media)


def test_no_implicit_decoder_install_or_header_only_as_decode(tmp_path, monkeypatch):
    profile, metadata, media = fixture(tmp_path)
    monkeypatch.setitem(sys.modules, "soundfile", None)
    assert (
        entry.verify(profile, metadata_root=metadata, media_root=media)[
            "full_decode_requested"
        ]
        is False
    )
    with pytest.raises(ModuleNotFoundError):
        entry.verify(profile, metadata_root=metadata, media_root=media, decode=True)


def test_original_duration_relation_is_checked(tmp_path):
    profile, metadata, media = fixture(tmp_path)
    path = metadata / "original.json"
    obj = json.loads(path.read_text())
    obj["data"][0]["duration"] = 2.0
    path.write_text(json.dumps(obj))
    profile["metadata"].update(
        bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="duration"):
        entry.verify(profile, metadata_root=metadata, media_root=media)


def test_pinned_profile_keeps_original_soundbible_ids_only():
    profile = json.loads(entry.PROFILE.read_text())
    assert [item["id"] for item in profile["files"]] == ["2219", "2218", "2217"]
    assert profile["component"] == "SoundBible"
    assert profile["archive_bytes"] == 579522592
    assert all(item["sample_rate"] == 32000 for item in profile["files"])
