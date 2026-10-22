import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from migrate_mod_git_evidence import verify  # noqa: E402

FOLDER = ROOT / "archive/mod-git-20261022/split-prose-source"


def test_original_document_bytes():
    verify(FOLDER)
    manifest = json.loads((FOLDER / "manifest.json").read_text())
    assert len(manifest["files"]) == 36
    assert manifest["source_visibility"] == "public"


def test_missing_probes_are_not_replaced_by_summary_links():
    audit = json.loads((FOLDER / "local-reference-audit.json").read_text())
    assert len(audit["references"]) == 39
    recovered = []
    for row in audit["references"]:
        assert (FOLDER / "snapshot" / row["source_file"]).is_file()
        assert not row["exists_at_original_path"]
        if row["status"] == "unrecovered-local-original":
            assert row["canonical_raw_url"] is None
        else:
            assert row["status"] == "recovered-by-declared-sha256"
            parts = urlparse(row["canonical_raw_url"]).path.split("/")
            assert parts[1:4] == ["vLLM-HUST", "vllm-hust-benchmark", "blob"]
            raw = (ROOT / "/".join(parts[5:])).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == row["declared_sha256"]
            recovered.append(row)
    assert len(recovered) == 2
