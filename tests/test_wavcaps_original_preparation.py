"""Preparation controls use invented compressed fixtures, not original audio proof."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import zlib

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec = importlib.util.spec_from_file_location(
        "wavcaps_prepare", ROOT / "scripts/prepare_wavcaps_original_media.py"
    )
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("component", list(entry.PROFILES))
def test_fixed_plan_is_offline_and_bounded(component):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_wavcaps_original_media.py"),
            "--component",
            component,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    profile = json.loads(entry.PROFILES[component].read_text())
    assert [item["id"] for item in report["plan"]] == [
        item["id"] for item in profile["files"]
    ]
    assert all(item["end"] - item["start"] < 8 * 1024**2 for item in report["plan"])
    assert report["model_calls"] == 0


def test_original_member_identity_and_trailing_data_rejected():
    raw = b"invented compressed fixture, not original FLAC"
    compressor = zlib.compressobj(wbits=-15)
    compressed = compressor.compress(raw) + compressor.flush()
    expected = {
        "bytes": len(raw),
        "crc32": zlib.crc32(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert entry.extract(compressed, expected) == raw
    for bad in (compressed[:-1], compressed + b"extra"):
        with pytest.raises(ValueError):
            entry.extract(bad, expected)
    with pytest.raises(ValueError):
        entry.extract(compressed, {**expected, "sha256": "0" * 64})


@pytest.mark.parametrize(
    "status,content_range", [(200, "bytes 1-3/10"), (206, "bytes 1-4/10")]
)
def test_no_whole_archive_or_wrong_range_response(status, content_range):
    class Response(io.BytesIO):
        pass

    response = Response(b"abc")
    response.status = status
    response.headers = {"Content-Range": content_range}
    with pytest.raises(ValueError, match="exact unencoded bounded range"):
        entry.fetch(
            {
                "url": "https://example.invalid/original",
                "start": 1,
                "end": 3,
                "part_bytes": 10,
            },
            opener=lambda *args, **kwargs: response,
        )
