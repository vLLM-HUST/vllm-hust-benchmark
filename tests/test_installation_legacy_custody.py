"""Byte custody for the bounded installation/legacy review, not admission."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "archive/mod-targeted-20261023"


def test_legacy_document_bytes():
    base = BASE / "vspec-agentx-school"
    manifest = json.loads((base / "manifest.json").read_text())
    assert len(manifest["files"]) == 4
    for row in manifest["files"]:
        raw = (base / "snapshot" / row["path"]).read_bytes()
        assert len(raw) == row["bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]


def test_installation_asset_review_and_manager_bindings():
    review = json.loads((BASE / "installation-consumers-20261024.json").read_text())
    assets = [a for r in review["repositories"] for a in r["assets"]]
    assert len(assets) == len({a["id"] for a in assets}) == 9
    for asset in assets:
        assert asset["bytes"] > 0
        assert len(asset["sha256"]) == 64
        assert asset["disposition"].startswith("retain-installation-asset;")
    sdist = next(a for a in assets if a["id"] == 629813551)
    bindings = sdist["evidence_bindings"]
    assert len(bindings) == 25
    assert sum(r["status"] == "existing-identical-original" for r in bindings) == 10
    for row in bindings:
        path = Path(row["benchmark_path"])
        assert not path.is_absolute() and ".." not in path.parts
        raw = (ROOT / path).read_bytes()
        assert len(raw) == row["bytes"]
        assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        if row["existing_url"]:
            assert row["existing_url"].endswith("/" + row["benchmark_path"])
