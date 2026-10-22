import io
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from archive_split_batch_release import PREFIX, members  # noqa: E402
from migrate_mod_git_evidence import verify  # noqa: E402


def tar(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, kind in entries:
            info = tarfile.TarInfo(name)
            info.type = kind
            archive.addfile(info, io.BytesIO())
    return stream.getvalue()


def test_frozen_members():
    verify(ROOT / "archive/mod-git-20261022/split-batch-release-v0.1.3")


@pytest.mark.parametrize(
    "path", ["/absolute", "../escape", "x/../escape", "x//ambiguous"]
)
def test_unsafe_path(path):
    with pytest.raises(ValueError):
        members(tar([(path, tarfile.REGTYPE)]))


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_special_member(kind):
    with pytest.raises(ValueError):
        members(tar([(PREFIX + "link", kind)]))


def test_duplicate():
    with pytest.raises(ValueError):
        members(tar([(PREFIX + "empty.json", tarfile.REGTYPE)] * 2))


def test_empty_original_preserved():
    assert members(tar([(PREFIX + "empty.json", tarfile.REGTYPE)])) == {
        PREFIX + "empty.json": b""
    }
