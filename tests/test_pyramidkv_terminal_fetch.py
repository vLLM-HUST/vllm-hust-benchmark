"""Retrieval integrity checks without public registry requests."""

import importlib.util
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
spec = importlib.util.spec_from_file_location(
    "audit_pyramidkv_terminal", SCRIPTS / "audit_pyramidkv_terminal.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
sys.modules[spec.name] = audit
spec = importlib.util.spec_from_file_location(
    "terminal_fetch", SCRIPTS / "fetch_pyramidkv_terminal.py"
)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def test_cached_file_checked_without_network(tmp_path, monkeypatch):
    target = tmp_path / "cached"
    target.write_bytes(b"original")
    monkeypatch.setattr(
        fetch, "get", lambda *a: pytest.fail("Unexpected network request")
    )
    assert fetch.save_checked(target, audit.sha(b"original"), "unused") == b"original"
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        fetch.save_checked(target, audit.sha(b"original"), "unused")


def test_download_mismatch_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "get", lambda *a: b"wrong")
    target = tmp_path / "new"
    with pytest.raises(ValueError, match="frozen digest"):
        fetch.save_checked(target, audit.sha(b"expected"), "unused")
    assert not target.exists()


@pytest.mark.parametrize("cross_host", [False, True])
def test_redirect_token_scope(cross_host):
    request = urllib.request.Request(
        "https://registry.example/a", headers={"Authorization": "Bearer synthetic"}
    )
    target = "https://cdn.example/b" if cross_host else "https://registry.example/b"
    redirected = fetch.PublicRedirect().redirect_request(
        request, None, 302, "Found", {}, target
    )
    assert redirected.has_header("Authorization") is (not cross_host)


def archive_fixture(
    tmp_path,
    name="root/tasks/example/instruction.md",
    *,
    symlink=False,
    duplicate=False,
):
    path = tmp_path / "snapshot.tar"
    with tarfile.open(path, "w") as archive:
        for _ in range(2 if duplicate else 1):
            info = tarfile.TarInfo(name)
            if symlink:
                info.type = tarfile.SYMTYPE
                info.linkname = "../../outside"
                archive.addfile(info)
            else:
                info.size = 9
                archive.addfile(info, io.BytesIO(b"synthetic"))
    return path


def test_extract_and_resume_preserves_data(tmp_path):
    archive = archive_fixture(tmp_path)
    destination = tmp_path / "output"
    fetch.extract_snapshot(archive, destination)
    fetch.extract_snapshot(archive, destination)
    target = destination / "tasks/example/instruction.md"
    assert target.read_bytes() == b"synthetic"
    target.write_bytes(b"local change")
    with pytest.raises(ValueError, match="replace"):
        fetch.extract_snapshot(archive, destination)
    assert target.read_bytes() == b"local change"


@pytest.mark.parametrize(
    "change", ["absolute", "traversal", "symlink", "duplicate", "destination_symlink"]
)
def test_extract_rejects_unsafe_archive(tmp_path, change):
    name = {"absolute": "/root/escape", "traversal": "root/../escape"}.get(
        change, "root/tasks/example/instruction.md"
    )
    archive = archive_fixture(
        tmp_path, name, symlink=change == "symlink", duplicate=change == "duplicate"
    )
    destination = tmp_path / "output"
    if change == "destination_symlink":
        (tmp_path / "outside").mkdir()
        destination.symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError):
        fetch.extract_snapshot(archive, destination)
