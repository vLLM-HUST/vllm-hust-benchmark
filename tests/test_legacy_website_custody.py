"""Verify every original byte in the versioned website custody archive."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/website-legacy-20261118"


def test_legacy_website_custody_bytes_and_member_inventory():
    manifest = json.loads((ARCHIVE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "website-legacy-custody/v1"
    assert manifest["source_repository"] == "vLLM-HUST/vllm-hust-website"
    pieces = []
    for part in manifest["parts"]:
        assert PurePosixPath(part["path"]).name == part["path"]
        raw = (ARCHIVE / part["path"]).read_bytes()
        assert len(raw) == part["bytes"] <= 900000
        assert hashlib.sha256(raw).hexdigest() == part["sha256"]
        pieces.append(raw)
    packed = b"".join(pieces)
    assert len(packed) == manifest["archive_bytes"]
    assert hashlib.sha256(packed).hexdigest() == manifest["archive_sha256"]
    expected = {entry["path"]: entry for entry in manifest["files"]}
    assert len(expected) == len(manifest["files"])
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r|gz") as archive:
        for member in archive:
            assert member.isfile() and member.name not in seen
            path = PurePosixPath(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            entry = expected[member.name]
            original = archive.extractfile(member).read()
            assert len(original) == entry["bytes"]
            assert hashlib.sha256(original).hexdigest() == entry["sha256"]
            header = b"blob " + str(len(original)).encode() + b"\0"
            assert (
                hashlib.sha1(header + original).hexdigest() == entry["source_git_blob"]
            )
            assert manifest["source_commit"] in entry["original_url"]
            seen.add(member.name)
    assert seen == set(expected)
