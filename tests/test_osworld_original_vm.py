"""Tiny archive/control fixtures, not a downloaded or booted OSWorld VM."""

import hashlib
import io
import json
from pathlib import Path
import stat
import struct
import sys
from types import SimpleNamespace
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_osworld_original_vm as module
import prepare_mlagentbench_cifar_assets as common


def header(backing=False):
    raw = bytearray(104)
    raw[:4] = b"QFI\xfb"
    struct.pack_into(">I", raw, 4, 3)
    struct.pack_into(">Q", raw, 24, 4096)
    if backing:
        struct.pack_into(">Q", raw, 8, 104)
        struct.pack_into(">I", raw, 16, 8)
    return bytes(raw)


def fixture(tmp_path, monkeypatch, entries=None):
    root = tmp_path / "assets"
    root.mkdir()
    archive = root / "Ubuntu.qcow2.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for name, payload in entries or [("Ubuntu.qcow2", header())]:
            info = zipfile.ZipInfo("fixture")
            # Preserve raw ZIP names even on Windows, whose constructor
            # normalizes backslashes before writing a string-name entry.
            info.filename = info.orig_filename = name
            output.writestr(info, payload)
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "revision": "fixture-not-publisher-revision",
                "asset": {
                    "path": archive.name,
                    "bytes": archive.stat().st_size,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "url": "https://example.invalid/fixture.zip",
                },
                "expected_vm_member": "Ubuntu.qcow2",
                "extracted_vm": {
                    "path": "Ubuntu.qcow2",
                    "bytes": len(header()),
                    "sha256": hashlib.sha256(header()).hexdigest(),
                    "identity_provenance": "explicit test fixture, not publisher data",
                },
            }
        )
    )
    monkeypatch.setattr(module, "PROFILE", profile)
    return SimpleNamespace(
        output=root,
        action="plan",
        max_bytes=None,
        resume=False,
        extract_output=tmp_path / "expanded",
        max_unpacked_bytes=None,
    )


def test_plan_has_no_acquisition_or_extraction(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    args.output = tmp_path / "unused-assets"
    result = module.run(args)
    assert (
        result["source_id"] == "OSWORLD" and result["source_checkbox_eligible"] is False
    )
    assert not args.extract_output.exists()
    assert not args.output.exists()


def test_inspect_and_extract_original_bytes_without_boot(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    args.action = "inspect"
    assert module.run(args)["qcow2"]["virtual_bytes"] == 4096
    assert not args.extract_output.exists()
    args.action, args.max_unpacked_bytes = "extract", 104
    report = module.run(args)
    assert (args.extract_output / "Ubuntu.qcow2").read_bytes() == header()
    assert report["all_members_crc_verified"] is True
    assert report["source_checkbox_eligible"] is False
    with pytest.raises(ValueError, match="fresh"):
        module.run(args)


def test_archive_and_unpacked_budgets_before_output_creation(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    args.action, args.max_bytes = "acquire", 0
    with pytest.raises(ValueError, match="complete original archive"):
        module.run(args)
    args.action, args.max_unpacked_bytes = "extract", 103
    with pytest.raises(ValueError, match="uncompressed"):
        module.run(args)
    assert not args.extract_output.exists()


def test_reuse_prepared_vm_without_archive_and_reject_mutated_image(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path, monkeypatch)
    args.action, args.max_unpacked_bytes = "extract", 104
    module.run(args)
    args.output = args.extract_output
    args.action = "verify-image"
    report = module.run(args)
    assert report["archive_reverified_this_call"] is False
    assert report["source_checkbox_eligible"] is False
    (args.output / "Ubuntu.qcow2").write_bytes(header() + b"mutation")
    with pytest.raises(ValueError, match="exact original"):
        module.run(args)


@pytest.mark.parametrize(
    "name",
    ["../escape", "/absolute", "C:/escape", "a\\escape", "ubuntu.qcow2", "bad\0suffix"],
)
def test_unsafe_or_case_duplicate_members_rejected(tmp_path, monkeypatch, name):
    args = fixture(
        tmp_path, monkeypatch, [("Ubuntu.qcow2", header()), (name, b"fixture")]
    )
    args.action = "extract"
    args.max_unpacked_bytes = 1024
    with pytest.raises(ValueError, match="unsafe"):
        module.run(args)
    assert not args.extract_output.exists()


def test_symlink_and_backing_file_rejected():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as output:
        output.writestr("Ubuntu.qcow2", header())
        link = zipfile.ZipInfo("link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(link, "Ubuntu.qcow2")
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
        with pytest.raises(ValueError, match="unsafe"):
            module.members(archive, "Ubuntu.qcow2")
    with pytest.raises(ValueError, match="backing-file"):
        module.qcow_header(header(backing=True))
    with pytest.raises(ValueError, match="QCOW2 version"):
        module.qcow_header(header()[:72])


def test_corrupt_archive_and_disk_budget_rejected(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    args.action, args.max_unpacked_bytes = "extract", 104
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="capacity"):
        module.run(args)
    assert not args.extract_output.exists()
    (args.output / "Ubuntu.qcow2.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="exact original"):
        module.run(args)


def test_common_acquisition_marker_preserves_source_identity(tmp_path):
    body = b"fixture-bytes"
    spec = {
        "path": "fixture.zip",
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "url": "https://example.invalid/fixture.zip",
    }

    def opener(request, timeout):
        response = io.BytesIO(body)
        response.status, response.headers = 200, {}
        return response

    for source in ["MLAGENTBENCH", "OSWORLD"]:
        root = tmp_path / source
        common.acquire(root, spec, max_bytes=len(body), opener=opener, source_id=source)
        saved = json.loads((root / "fixture.zip.acquisition.private.json").read_text())
        assert saved["source_id"] == source
        assert (root / "fixture.zip").read_bytes() == body
