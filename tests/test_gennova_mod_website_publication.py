"""The public MOD test report is complete, bounded, and free of host details."""

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_gennova_mod_website_publication.py"
SPEC = importlib.util.spec_from_file_location("mod_publication", SCRIPT)
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)
OUTPUT = ROOT / "website-publication/data/mod-test-evidence.json"


def test_publication_is_current_and_complete():
    assert publication.main(["--check"]) == 0
    payload = json.loads(OUTPUT.read_text())
    assert payload["schema_version"] == "mod-test-evidence/v1"
    assert payload["summary"]["catalog_entries"] == 107
    assert len(payload["integration_records"]) == 55
    assert len(payload["combination_effects"]) == 29
    assert len(payload["negative_results"]) == 90
    assert payload["scope"]["website_report"] is True
    assert payload["scope"]["official_leaderboard_submission"] is False
    assert payload["scope"]["new_measurements"] is False


def test_publication_omits_private_machine_and_temporary_details():
    raw = OUTPUT.read_text()
    forbidden = (
        r"/(?:tmp|root|home)/",
        r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        r"\b192\.168\.\d{1,3}\.\d{1,3}\b",
        r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b",
        r"github[_-]?token",
        r"authorization:\s*bearer",
    )
    assert not any(re.search(pattern, raw, re.IGNORECASE) for pattern in forbidden)
