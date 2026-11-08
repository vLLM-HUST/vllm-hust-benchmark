import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "swesmith_recovery",
    Path(__file__).resolve().parents[1]
    / "scripts/recover_swesmith_fixed3_repository.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_git_object_known_empty_blob():
    assert (
        MODULE.git_object("blob", b"")[0]
        == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"  # pragma: allowlist secret - standard Git empty-blob object ID
    )


def test_archive_rejects_parent_escape(tmp_path):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("root/../outside")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        MODULE.archive_objects(archive)


@pytest.mark.skipif(
    not shutil.which("git"), reason="Git required for object verification"
)
@pytest.mark.parametrize("include_parent", [False, True])
def test_exact_shallow_recovery_without_network(tmp_path, monkeypatch, include_parent):
    archive = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("fixture/hello.txt")
        info.mode = 0o644
        info.size = 6
        tar.addfile(info, io.BytesIO(b"hello\n"))
    tree, _ = MODULE.archive_objects(archive)
    raw = (
        f"tree {tree}\nauthor Fixture <fixture@example.invalid> 0 +0000\n"
        "committer Fixture <fixture@example.invalid> 0 +0000\n\nfixture\n"
    ).encode()
    parent_raw = raw.replace(b"\n\nfixture\n", b"\n\nparent\n")
    parent = MODULE.git_object("commit", parent_raw)[0]
    if include_parent:
        raw = raw.replace(
            b"\nauthor ", b"\nparent " + parent.encode() + b"\nauthor ", 1
        )
        (tmp_path / (parent + ".tar.gz")).write_bytes(archive.read_bytes())
    commit = MODULE.git_object("commit", raw)[0]
    archive.rename(tmp_path / (commit + ".tar.gz"))
    person = {
        "name": "Fixture",
        "email": "fixture@example.invalid",
        "date": "1970-01-01T00:00:00Z",
    }
    metadata = {
        "tree": {"sha": tree},
        "parents": [{"sha": parent}] if include_parent else [],
        "author": person,
        "committer": person,
        "message": "fixture",
        "verification": {"signature": None},
    }
    monkeypatch.setattr(MODULE, "COMMITS", {"fixture-task": commit})
    parent_metadata = {**metadata, "parents": [], "message": "parent"}
    monkeypatch.setattr(
        MODULE,
        "urlopen",
        lambda request, **kwargs: io.BytesIO(
            json.dumps(
                parent_metadata if request.full_url.endswith(parent) else metadata
            ).encode()
        ),
    )
    output = tmp_path / "recovered"
    report = MODULE.recover(tmp_path, output, include_grading_parents=include_parent)
    assert (output / "hello.txt").read_bytes() == b"hello\n"
    actual = subprocess.check_output(
        ["git", "-C", str(output), "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual == commit
    if include_parent:
        assert (
            subprocess.check_output(
                ["git", "-C", str(output), "rev-parse", "HEAD~1"], text=True
            ).strip()
            == parent
        )
    assert report["source_checkbox_eligible"] is False and report["model_calls"] == 0
    with pytest.raises(ValueError, match="original commit ID"):
        MODULE.commit_bytes({**metadata, "message": "changed"}, commit)
