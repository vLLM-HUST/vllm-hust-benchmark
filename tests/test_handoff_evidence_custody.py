"""Retained reports must keep the original evidence bytes through integration."""

import hashlib
import json
from pathlib import Path


def test_retained_evidence_matches_original_git_blobs():
    root = Path(__file__).resolve().parents[1]
    for report in (
        "retained-evidence-integration-20261104",
        "handoff-reconciliation-20261104",
    ):
        manifest = json.loads((root / "reports" / report / "sources.json").read_text())
        checked = 0
        for entry in manifest["sources"]:
            if entry["treatment"] != "byte-preserved":
                continue
            raw = (root / entry["path"]).read_bytes()
            identity = hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest()
            assert identity == entry["source_git_blob"], entry["path"]
            checked += 1
        assert checked > 0
