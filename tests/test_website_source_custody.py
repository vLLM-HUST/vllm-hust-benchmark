import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from migrate_mod_git_evidence import verify  # noqa: E402


def test_public_website_source_custody():
    archives = sorted((ROOT / "archive/mod-git-20261022").glob("website-source-*"))
    assert len(archives) == 8
    count = 0
    for archive in archives:
        verify(archive)
        manifest = json.loads((archive / "manifest.json").read_text())
        assert manifest["source_visibility"] == "public"
        count += len(manifest["files"])
    assert count == 48
