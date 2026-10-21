import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "next_mod_custody", ROOT / "scripts/migrate_mod_git_evidence.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("repository", ["gdn-state-codec", "vllm-hust-opset"])
def test_frozen_originals(repository):
    directory = ROOT / "archive/mod-git-20261022" / repository
    MODULE.verify(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["source_visibility"] == "public"
    assert manifest["files"]
    assert len(manifest["source_commit"]) == 40


@pytest.mark.parametrize("prefix", ["../", "/", "", "results"])
def test_unsafe_selection_rejected_before_network(tmp_path, prefix):
    with pytest.raises(ValueError):
        MODULE.prepare(tmp_path, "unused", tmp_path / "out", prefixes=(prefix,))
