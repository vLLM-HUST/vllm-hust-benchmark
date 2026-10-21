import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "archive/mod-releases-20261021"


def test_kv_cleanup_preserves_verified_custody_and_historical_resolver():
    cleanup = json.loads((ROOT / "kv-arrival-cleanup.json").read_text())
    custody = json.loads((ROOT / "kv-arrival.json").read_text())
    assets = [a for r in custody["releases"] for a in r["assets"]]
    assert len(assets) == 17
    assert cleanup["verified_assets"] == assets
    assert set(cleanup["removed_assets"]) == {a["source_id"] for a in assets}
    assert "scripts/canonical_evidence.py" in cleanup["retained_historical_references"]
    assert all(a["download_back_verified"] for a in assets)


def test_muxpd_cleanup_is_scoped_to_four_migrated_assets():
    cleanup = json.loads((ROOT / "muxpd-cleanup.json").read_text())
    assert len(cleanup["removed_assets"]) == 4
    assert sum(a["bytes"] for a in cleanup["removed_assets"]) == 110153887
    assert cleanup["archive_member_verification"] == {"goodput": 763, "groupfx": 348}
