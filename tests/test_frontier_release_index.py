import gzip
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "release_index",
    Path(__file__).resolve().parents[1] / "scripts/index_frontier_release.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def bundle(name="./failed/status.json"):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        raw = json.dumps({"passed": False, "stage": "qualification"}).encode()
        member = tarfile.TarInfo(name)
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    return gzip.compress(buffer.getvalue(), mtime=0)


def test_failed_attempt_is_preserved():
    raw = bundle()
    files, runs = MODULE.index_archive(raw, MODULE.sha(raw))
    assert len(files) == 1
    assert runs[0]["original_status"] == {"passed": False, "stage": "qualification"}


def test_archive_hash_must_match():
    with pytest.raises(ValueError, match="digest"):
        MODULE.index_archive(bundle(), "0" * 64)


def test_path_traversal_rejected():
    raw = bundle("../bad/status.json")
    with pytest.raises(ValueError, match="Unsafe"):
        MODULE.index_archive(raw, MODULE.sha(raw))
