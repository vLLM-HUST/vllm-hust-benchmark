import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from archive_split_batch_history import members  # noqa: E402
from migrate_mod_git_evidence import verify  # noqa: E402


@pytest.mark.parametrize(
    "version,count", [("0.1.0", 350), ("0.1.1", 350), ("0.1.2", 359), ("0.1.3", 284)]
)
def test_original_release_members(version, count):
    folder = ROOT / "archive/mod-git-20261022" / ("split-history-release-" + version)
    verify(folder)
    receipt = json.loads((folder / "manifest.json").read_text())
    assert len(receipt["files"]) == count
    assert receipt["source_visibility"] == "public"


def test_current_git_originals():
    verify(ROOT / "archive/mod-git-20261022/split-current-git")


def test_migration_receipt_matches_frozen_originals():
    folder = ROOT / "archive/mod-git-20261022"
    receipt = json.loads((folder / "split-history-migration.json").read_text())
    manifest = json.loads((folder / "split-current-git/manifest.json").read_text())
    records = {row["path"]: row for row in manifest["files"]}
    removed = receipt["removed_git_copies"]
    retained = receipt["retained_source_paths"]
    assert len(removed) == 374
    assert len(retained) == 64
    assert not {r["path"] for r in removed} & {r["path"] for r in retained}
    for row in removed + retained:
        assert row["sha256"] == records[row["path"]]["sha256"]
    audit = receipt["installation_consumer_audit"]
    assert audit["deleted_release_assets"] == []
    assert len(audit["releases"]) == 4
    for release in audit["releases"]:
        assert len(release["assets"]) == 3
        distributions = [a for a in release["assets"] if a["name"] != "dist.sha256"]
        assert all(a["pypi_digest_identity_verified"] for a in distributions)
    assert receipt["remote_verification"]["files"] == 1833
    assert receipt["remote_verification"]["bytes"] == 45789686


@pytest.mark.parametrize(
    "name,kind",
    [
        ("root/../escape", tarfile.REGTYPE),
        ("wrong/results/a", tarfile.REGTYPE),
        ("root/results/link", tarfile.SYMTYPE),
    ],
)
def test_unsafe_member(name, kind):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        archive.addfile(member, io.BytesIO())
    with pytest.raises(ValueError):
        members(stream.getvalue(), "root", ("results/",))
