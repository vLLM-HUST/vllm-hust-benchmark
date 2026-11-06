"""Migrated RIDE originals remain byte-identical, including Markdown snapshots."""

import hashlib
import json
from pathlib import Path


def test_ride_custody_original_bytes():
    root = Path(__file__).resolve().parents[1] / "reports/ride-control-plane-20261106"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["new_experiments"] is False
    assert manifest["private_raw_archives_uploaded"] is False
    assert len(manifest["files"]) == 6
    for record in manifest["files"]:
        data = (root / record["path"]).read_bytes()
        assert len(data) == record["bytes"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
