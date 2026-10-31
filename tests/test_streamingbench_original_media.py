"""Invented ZIP fixtures only; no original dataset, video or model execution."""

import importlib.util
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

spec = importlib.util.spec_from_file_location(
    "stream_media",
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_streamingbench_original_media.py",
)
runner = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(spec.origin).parent))
try:
    spec.loader.exec_module(runner)
finally:
    sys.path.pop(0)


def fixture(descriptor=False):
    payload = b"tiny invented member bytes" * 100

    class StreamingOutput(io.BytesIO):
        def seek(self, *args):
            raise io.UnsupportedOperation("nonseekable fixture")

    buffer = StreamingOutput() if descriptor else io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sample_1/video.mp4", payload)
        archive.writestr("padding", b"x" * 1024, compress_type=zipfile.ZIP_STORED)
    raw = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        item = archive.getinfo("sample_1/video.mp4")
    member = {
        "method": item.compress_type,
        "flags": item.flag_bits,
        "crc32": item.CRC,
        "bytes": item.file_size,
        "compressed_bytes": item.compress_size,
        "archive_member": item.filename,
        "local_header_offset": item.header_offset,
        "original_task_media_path": "./videos/sample_1_real.mp4",
    }
    archive = {
        "archive": "Original.zip",
        "bytes": len(raw),
        "captured_etag": '"original"',
    }
    inventory = {"dataset_repo": "fixture/fixture", "dataset_revision": "fixed"}

    def opener(request, *, timeout):
        start, end = map(
            int, request.get_header("Range").removeprefix("bytes=").split("-")
        )
        response = io.BytesIO(raw[start : end + 1])
        response.status = 206
        response.headers = {
            "ETag": '"original"',
            "Content-Range": f"bytes {start}-{end}/{len(raw)}",
        }
        return response

    return payload, member, archive, inventory, opener


@pytest.mark.parametrize("descriptor", [False, True])
def test_exact_original_member_stream_and_crc(descriptor):
    payload, member, archive, inventory, opener = fixture(descriptor)
    sink = io.BytesIO()
    report = runner.transfer(member, archive, inventory, sink, opener=opener)
    assert sink.getvalue() == payload
    assert report["original_complete_member_verified"]


def test_changed_archive_or_crc_is_rejected():
    _, member, archive, inventory, opener = fixture()
    with pytest.raises(ValueError, match="ETag"):
        runner.transfer(
            member,
            {**archive, "captured_etag": '"changed"'},
            inventory,
            io.BytesIO(),
            opener=opener,
        )
    with pytest.raises(ValueError, match="CRC"):
        runner.transfer(
            {**member, "crc32": member["crc32"] + 1},
            archive,
            inventory,
            io.BytesIO(),
            opener=opener,
        )


def test_capacity_guard_precedes_network_and_existing_files_are_not_overwritten(
    tmp_path, monkeypatch
):
    _, member, archive, inventory, _ = fixture()
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    monkeypatch.setattr(
        runner, "transfer", lambda *a, **kw: pytest.fail("network must not run")
    )
    with pytest.raises(ValueError, match="insufficient space"):
        runner.download(member, archive, inventory, tmp_path, max_bytes=member["bytes"])
    target = tmp_path / "sample_1_real.mp4"
    target.write_bytes(b"keep")
    with pytest.raises(ValueError, match="never overwrite"):
        runner.download(member, archive, inventory, tmp_path, max_bytes=member["bytes"])
    assert target.read_bytes() == b"keep"
