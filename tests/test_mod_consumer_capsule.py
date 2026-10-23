import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_mod_consumer_evidence.py"
SPEC = importlib.util.spec_from_file_location("capsule", SCRIPT)
capsule = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capsule)


def bundle(
    name="snapshot/data.json", content=b"failure retained", link=False, duplicate=False
):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        if link:
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            member.size = 0
        archive.addfile(member, io.BytesIO(content))
        if duplicate:
            archive.addfile(member, io.BytesIO(content))
    raw = stream.getvalue()
    manifest = {
        "archive": {
            "bytes": len(raw),
            "sha256": capsule.digest(raw),
            "member_prefix": "snapshot/",
        },
        "files": [
            {
                "path": "data.json",
                "bytes": len(content),
                "sha256": capsule.digest(content),
            }
        ],
    }
    return manifest, raw


def test_preserves_original_bytes():
    manifest, raw = bundle()
    assert capsule.verify(manifest, raw) == {"data.json": b"failure retained"}


@pytest.mark.parametrize(
    "kwargs", [{"name": "snapshot/../escape"}, {"link": True}, {"duplicate": True}]
)
def test_unsafe_members_rejected(kwargs):
    manifest, raw = bundle(**kwargs)
    with pytest.raises(ValueError):
        capsule.verify(manifest, raw)


def test_hash_missing_and_unsafe_manifest_rejected():
    manifest, raw = bundle()
    with pytest.raises(ValueError):
        capsule.verify(manifest, raw + b"changed")
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        capsule.verify(manifest, raw)
    manifest["files"][0]["path"] = "../escape"
    with pytest.raises(ValueError):
        capsule.verify(manifest, raw)
