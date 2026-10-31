"""Invented HTTP/TAR control fixtures, not publisher video or inference proof."""

from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location(
        "ovo_capture", ROOT / "scripts/capture_ovobench_original_tar_headers.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.pop(0)

INVENTORY = {
    "dataset_repo": "fixture/source",
    "dataset_revision": "fixture",
    "parts": [{"path": "chunked_videos.a", "bytes": 10000}],
}


def response(raw, start, etag='"original"'):
    stream = io.BytesIO(raw)
    stream.status = 206
    stream.headers = {
        "ETag": etag,
        "Content-Range": f"bytes {start}-{start + 511}/10000",
    }
    return stream


def test_only_headers_are_read_and_all_payload_bytes_skipped():
    directory = tarfile.TarInfo("chunked_videos")
    directory.type = tarfile.DIRTYPE
    media = tarfile.TarInfo("chunked_videos/0.mp4")
    media.size = 700
    headers = {
        0: directory.tobuf(),
        512: media.tobuf(),
        2048: bytes(512),
        2560: bytes(512),
    }
    calls = []

    def opener(request, **kwargs):
        start = int(request.get_header("Range").split("=")[1].split("-")[0])
        calls.append(start)
        return response(headers[start], start)

    state = entry.empty_index()
    for _ in range(4):
        state = entry.advance(state, INVENTORY, opener=opener)
    assert calls == [0, 512, 2048, 2560]  # Never read the 700-byte member payload.
    assert len(state["entries"]) == 2 and state["complete"] is True
    assert state["zero_headers"] == 2 and state["next_offset"] == 3072
    assert (
        entry.advance(
            state,
            INVENTORY,
            opener=lambda *a, **k: pytest.fail("complete index refetched"),
        )
        == state
    )


def test_failed_read_does_not_mutate_last_valid_checkpoint():
    state = entry.empty_index()
    state["etags"]["chunked_videos.a"] = '"expected"'
    before = deepcopy(state)
    with pytest.raises(ValueError, match="captured ETag"):
        entry.advance(state, INVENTORY, opener=lambda *a, **k: response(bytes(512), 0))
    assert state == before


def test_unsupported_original_member_is_not_silently_skipped():
    link = tarfile.TarInfo("unsafe-link")
    link.type = tarfile.SYMTYPE
    with pytest.raises(ValueError, match="unsupported/unsafe/duplicate"):
        entry.advance(
            entry.empty_index(),
            INVENTORY,
            opener=lambda *a, **k: response(link.tobuf(), 0),
        )


def test_owned_checkpoint_replacement_preserves_valid_json(tmp_path):
    state = entry.empty_index()
    entry.save_owned_checkpoint(tmp_path, state)
    assert json.loads((tmp_path / "checkpoint-private.json").read_text()) == state
    assert not (tmp_path / "checkpoint-next.private.json").exists()
    entry.save_owned_checkpoint(tmp_path, state)
