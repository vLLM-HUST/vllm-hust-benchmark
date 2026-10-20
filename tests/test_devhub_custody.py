import gzip
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/migrate_devhub_evidence.py"
SPEC = importlib.util.spec_from_file_location("devhub_custody", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    raw = b'{"passed": false}\n'
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        member = tarfile.TarInfo("results/failed.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    packed = gzip.compress(buffer.getvalue(), mtime=0)
    manifest = {
        "asset": {"sha256": MODULE.sha(packed), "bytes": len(packed)},
        "files": [
            {
                "path": "results/failed.json",
                "sha256": MODULE.sha(raw),
                "bytes": len(raw),
                "destination": {
                    "kind": "release-member",
                    "member": "results/failed.json",
                },
            }
        ],
    }
    return manifest, packed


def test_preserves_failed_result():
    manifest, packed = fixture()
    MODULE.verify(manifest, packed, Path("."))


def test_rejects_corrupt_archive():
    manifest, packed = fixture()
    with pytest.raises(ValueError, match="SHA256"):
        MODULE.verify(manifest, packed + b"x", Path("."))


def test_rejects_wrong_member_hash():
    manifest, packed = fixture()
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Evidence mismatch"):
        MODULE.verify(manifest, packed, Path("."))


def test_rejects_missing_member():
    manifest, packed = fixture()
    manifest["files"] = []
    with pytest.raises(ValueError, match="member set"):
        MODULE.verify(manifest, packed, Path("."))
