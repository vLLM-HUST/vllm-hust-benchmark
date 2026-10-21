import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "mod_git_custody", ROOT / "scripts/migrate_mod_git_evidence.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "directory", sorted((ROOT / "archive/mod-git-20261021").glob("stateaxis-*"))
)
def test_stateaxis_original_bytes(directory):
    MODULE.verify(directory)
    receipt = json.loads((directory / "manifest.json").read_text())
    assert receipt["source_visibility"] == "public"
    assert len(receipt["source_commit"]) == 40
    assert any(r["path"].startswith("evidence/") for r in receipt["files"])
    if directory.name == "stateaxis-pipelined-weight-loading":
        assert len(receipt["invalid_json_paths"]) == 14
        assert all(
            (directory / "snapshot" / p).read_bytes() == b""
            for p in receipt["invalid_json_paths"]
        )


def test_corruption_and_missing_members_are_rejected(tmp_path):
    source = next((ROOT / "archive/mod-git-20261021").glob("stateaxis-*/manifest.json"))
    receipt = json.loads(source.read_text())
    receipt["files"] = receipt["files"][:1]
    (tmp_path / "manifest.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Missing or unindexed"):
        MODULE.verify(tmp_path)
    member = tmp_path / "snapshot" / receipt["files"][0]["path"]
    member.parent.mkdir(parents=True)
    member.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity"):
        MODULE.verify(tmp_path)
