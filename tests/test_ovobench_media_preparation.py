"""Invented HTTP/file fixtures; actual original network proof is separate."""

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location(
        "ovo_media_prepare", ROOT / "scripts/prepare_ovobench_original_media.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.pop(0)


def fixture():
    raw = b"invented complete member, not real video"
    member = {
        "path": "0.mp4",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "archive_member": "chunked_videos/0.mp4",
    }
    location = {**member, "ranges": [{"part": "p", "start": 20, "end": 19 + len(raw)}]}
    inventory = {
        "dataset_repo": "fixture/source",
        "dataset_revision": "fixture",
        "parts": [{"path": "p", "bytes": 1000}],
    }
    return raw, member, location, inventory


def opener_for(raw, etag='"original"'):
    def opener(request, **kwargs):
        response = io.BytesIO(raw)
        response.status = 206
        response.headers = {
            "ETag": etag,
            "Content-Range": f"bytes 20-{19 + len(raw)}/1000",
        }
        return response

    return opener


def test_original_ranges_and_whole_member_identity_are_both_required():
    raw, member, location, inventory = fixture()
    sink = io.BytesIO()
    result = entry.transfer(
        member, location, inventory, {"p": '"original"'}, sink, opener=opener_for(raw)
    )
    assert sink.getvalue() == raw and result["sha256"] == member["sha256"]
    with pytest.raises(ValueError, match="captured original ETag"):
        entry.transfer(
            member,
            location,
            inventory,
            {"p": '"different"'},
            io.BytesIO(),
            opener=opener_for(raw),
        )
    with pytest.raises(ValueError, match="complete original member identity"):
        entry.transfer(
            {**member, "sha256": "0" * 64},
            location,
            inventory,
            {"p": '"original"'},
            io.BytesIO(),
            opener=opener_for(raw),
        )


def test_existing_original_is_reused_without_network_or_overwrite(
    tmp_path, monkeypatch
):
    raw, member, location, inventory = fixture()
    path = tmp_path / member["path"]
    path.write_bytes(raw)
    monkeypatch.setattr(
        entry,
        "transfer",
        lambda *args, **kwargs: pytest.fail("existing asset must not be refetched"),
    )
    result = entry.download(
        member, location, inventory, {}, tmp_path, max_bytes=len(raw)
    )
    assert result["existing_original_reused"] and not result["network_fetch_executed"]
    path.write_bytes(b"incompatible user file")
    with pytest.raises(ValueError, match="asset identity mismatch"):
        entry.download(member, location, inventory, {}, tmp_path, max_bytes=len(raw))
    assert path.read_bytes() == b"incompatible user file"


def test_download_requires_full_budget_and_keeps_existing_partial(tmp_path):
    raw, member, location, inventory = fixture()
    with pytest.raises(ValueError, match="byte budget"):
        entry.download(
            member, location, inventory, {}, tmp_path, max_bytes=len(raw) - 1
        )
    partial = tmp_path / "0.mp4.partial"
    partial.write_bytes(b"prior partial")
    with pytest.raises(ValueError, match="existing partial"):
        entry.download(member, location, inventory, {}, tmp_path, max_bytes=len(raw))
    assert partial.read_bytes() == b"prior partial"


def test_fresh_target_is_published_only_after_full_identity_passes(
    tmp_path, monkeypatch
):
    raw, member, location, inventory = fixture()
    original = entry.transfer
    monkeypatch.setattr(
        entry.shutil, "disk_usage", lambda _: SimpleNamespace(free=2**30)
    )
    monkeypatch.setattr(
        entry,
        "transfer",
        lambda member, location, inventory, etags, sink: original(
            member, location, inventory, etags, sink, opener=opener_for(raw)
        ),
    )
    good = tmp_path / "good"
    result = entry.download(
        member, location, inventory, {"p": '"original"'}, good, max_bytes=len(raw)
    )
    assert result["network_fetch_executed"] and (good / "0.mp4").read_bytes() == raw
    assert not (good / "0.mp4.partial").exists()
    bad = tmp_path / "bad"
    with pytest.raises(ValueError, match="complete original member identity"):
        entry.download(
            {**member, "sha256": "0" * 64},
            location,
            inventory,
            {"p": '"original"'},
            bad,
            max_bytes=len(raw),
        )
    assert not (bad / "0.mp4").exists()
    assert (bad / "0.mp4.partial").read_bytes() == raw
