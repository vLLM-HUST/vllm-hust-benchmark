import hashlib
import json
from pathlib import Path


def test_preserved_pipeline_files_match_manifest():
    root = Path(__file__).resolve().parents[1] / "archive/dev-hub-pipeline-20261020"
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["files"]) == 2
    for record in manifest["files"]:
        raw = (root / record["destination_path"]).read_bytes()
        assert len(raw) == record["bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        json.loads(raw)
