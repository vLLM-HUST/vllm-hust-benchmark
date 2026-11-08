"""Tiny invented ZIP controls; no QReCC corpus, network or models."""

import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
specification = importlib.util.spec_from_file_location(
    "qrecc_collection", ROOT / "scripts/prepare_qrecc_original_collection.py"
)
module = importlib.util.module_from_spec(specification)
specification.loader.exec_module(module)


@pytest.fixture
def original(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "DISK_RESERVE", 0)
    archive = tmp_path / "fixture.zip"
    raw = [
        b'{"id":"fixture-a","contents":"alpha"}\n',
        b'{"id":"fixture-b","contents":"beta"}\n',
    ]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("first/", b"")
        z.writestr("first/a.jsonl", raw[0])
        z.writestr("second/", b"")
        z.writestr("second/b.jsonl", raw[1])
    spec = {
        "bytes": archive.stat().st_size,
        "md5": hashlib.md5(archive.read_bytes()).hexdigest(),
        "entries": 4,
        "files": 2,
        "uncompressed_bytes": sum(map(len, raw)),
        "maximum_member_bytes": max(map(len, raw)),
    }
    return archive, spec, raw


def test_default_plan_has_no_side_effects():
    out = subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/prepare_qrecc_original_collection.py")],
        text=True,
    )
    report = json.loads(out)
    assert report["status"] == "PLAN_ONLY"
    assert report["archive"]["files"] == 56003
    assert not report["source_checkbox_eligible"]


def test_verify_is_readonly_and_does_not_claim_crc(original, tmp_path):
    archive, spec, _ = original
    before = archive.read_bytes()
    report = module.prepare(archive, spec=spec)
    assert report["publisher_md5_verified"] and not report["member_crc_verified"]
    assert archive.read_bytes() == before
    assert set(tmp_path.iterdir()) == {archive}


def test_extract_preserves_all_member_bytes_and_separates_metadata(original, tmp_path):
    archive, spec, raw = original
    output = tmp_path / "prepared"
    report = module.prepare(
        archive, output=output, max_bytes=spec["uncompressed_bytes"], spec=spec
    )
    assert report["completed_files"] == 2 and report["member_crc_verified"]
    assert (output / "collection/first/a.jsonl").read_bytes() == raw[0]
    assert (output / "collection/second/b.jsonl").read_bytes() == raw[1]
    assert len(list((output / "collection").rglob("*.*"))) == 2
    assert (output / "collection-preparation.private.json").is_file()
    assert not report["index_built"] and report["native_tasks_executed"] == 0
    with pytest.raises(ValueError, match="fresh output"):
        module.prepare(
            archive, output=output, max_bytes=spec["uncompressed_bytes"], spec=spec
        )


@pytest.mark.parametrize(
    "change,error",
    [
        ("bytes", "byte count"),
        ("md5", "checksum"),
        ("files", "inventory"),
        ("uncompressed_bytes", "inventory"),
    ],
)
def test_incomplete_or_changed_collection_rejected(original, change, error):
    archive, spec, _ = original
    spec = dict(spec)
    spec[change] = "0" * 32 if change == "md5" else spec[change] + 1
    with pytest.raises(ValueError, match=error):
        module.prepare(archive, spec=spec)


def test_explicit_whole_collection_budget_and_external_output(original):
    archive, spec, _ = original
    with pytest.raises(ValueError, match="budget"):
        module.prepare(archive, output=archive.parent / "missing", spec=spec)
    with pytest.raises(ValueError, match="outside checkout"):
        module.prepare(archive, output=ROOT / "unused-collection-fixture", spec=spec)


def test_other_writers_consuming_space_stop_before_next_member(
    original, tmp_path, monkeypatch
):
    archive, spec, _ = original
    spaces = iter(
        [SimpleNamespace(free=spec["uncompressed_bytes"]), SimpleNamespace(free=0)]
    )
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: next(spaces))
    output = tmp_path / "space-failure"
    with pytest.raises(ValueError, match="shared disk reserve"):
        module.prepare(
            archive, output=output, max_bytes=spec["uncompressed_bytes"], spec=spec
        )
    report = json.loads((output / "collection-preparation.private.json").read_text())
    assert report["completed_files"] == 0 and not report["member_crc_verified"]


@pytest.mark.parametrize(
    "name",
    ["../escape.jsonl", "/absolute.jsonl", "a\\b.jsonl", "a/./b.jsonl", "C:evil.jsonl"],
)
def test_unsafe_member_paths_rejected(tmp_path, name):
    info = zipfile.ZipInfo("fixture.jsonl")
    # Windows ZIP writers normalize backslashes; exercise raw central metadata.
    info.filename = info.orig_filename = name
    with pytest.raises(ValueError, match="unsafe"):
        module.members(SimpleNamespace(infolist=lambda: [info]), {})


def test_crc_failure_preserves_failed_output_and_no_success(original, tmp_path):
    archive, spec, _ = original
    raw = bytearray(archive.read_bytes())
    start = raw.index(b"PK\x01\x02")
    # First central record is a directory; corrupt the next file CRC only.
    start = raw.index(b"PK\x01\x02", start + 4)
    raw[start + 16] ^= 1
    archive.write_bytes(raw)
    spec = dict(spec, md5=hashlib.md5(raw).hexdigest())
    output = tmp_path / "failed-output"
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        module.prepare(
            archive, output=output, max_bytes=spec["uncompressed_bytes"], spec=spec
        )
    report = json.loads((output / "collection-preparation.private.json").read_text())
    assert report["status"] == "FAILED_EXTRACTION_RETAINED_NOT_INDEXED"
    assert not report["member_crc_verified"]


def test_symlink_member_rejected(tmp_path):
    archive = tmp_path / "symlink.zip"
    info = zipfile.ZipInfo("alias.jsonl")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(info, "outside")
    with zipfile.ZipFile(archive) as z, pytest.raises(ValueError, match="unsafe"):
        module.members(z, {})
