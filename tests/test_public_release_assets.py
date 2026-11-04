import hashlib
import io
import runpy
import tarfile
from pathlib import Path

import pytest

AUDIT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/verify_public_release_assets.py")
)


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("raw", [b"", b"original bytes\x00\xff", b"x" * 2_000_000])
def test_stream_verifies_exact_bytes(raw):
    target = io.BytesIO()
    assert AUDIT["check_stream"](io.BytesIO(raw), len(raw), digest(raw), target) == len(
        raw
    )
    assert target.getvalue() == raw


@pytest.mark.parametrize(
    "raw,size,expected",
    [
        (b"abc", 4, digest(b"abcd")),
        (b"abcd", 3, digest(b"abc")),
        (b"abc", 3, digest(b"abd")),
        (b"abc", 3, None),
    ],
)
def test_truncated_extended_or_tampered_assets_fail(raw, size, expected):
    with pytest.raises(ValueError):
        AUDIT["check_stream"](io.BytesIO(raw), size, expected)


def capsule(names):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 3
            archive.addfile(member, io.BytesIO(b"abc"))
    output.seek(0)
    return output


def test_archive_member_identity_and_completeness():
    expected = {"a": {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}}
    assert AUDIT["check_members"](capsule(["a"]), expected) == 1
    for names in [[], ["a", "a"], ["b"], ["../a"]]:
        with pytest.raises(ValueError):
            AUDIT["check_members"](capsule(names), expected)


def test_drafts_are_excluded_before_asset_listing(monkeypatch):
    inventory = AUDIT["inventory"]
    calls = []

    def pages(path):
        calls.append(path)
        if path.endswith("/releases"):
            return [{"draft": True, "id": 1, "tag_name": "private"}]
        raise AssertionError("Draft assets must not be requested")

    monkeypatch.setitem(inventory.__globals__, "pages", pages)
    assert inventory() == ([], [])
    assert len(calls) == 2
