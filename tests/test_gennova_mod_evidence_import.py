"""The GenNova MOD evidence import stays pinned, complete and non-publishing."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gennova_import", ROOT / "scripts/build_gennova_mod_evidence_import.py"
)
gennova_import = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gennova_import)


def test_generated_import_is_current():
    assert gennova_import.main(["--check"]) == 0


def test_manifest_inventory_and_boundaries():
    manifest = json.loads(
        (ROOT / "reports/gennova-mod-evidence-20261105/manifest.json").read_text()
    )
    assert manifest["source"]["commit"] == gennova_import.SOURCE_COMMIT
    assert len(manifest["catalog_entries"]) == 107
    assert len({item["id"] for item in manifest["catalog_entries"]}) == 107
    assert all(
        gennova_import.SOURCE_COMMIT in item["source_url"]
        for item in manifest["catalog_entries"]
    )
    assert manifest["scope"]["is_official_leaderboard_submission"] is False
    assert manifest["scope"]["contains_new_measurements"] is False
    assert manifest["scope"]["website_publication_enabled"] is False

    stats = {
        Path(item["snapshot_path"]).name: item["statistics"]
        for item in manifest["snapshots"]
    }
    assert stats["results-catalog.json"]["entry_count"] == 107
    assert stats["gennova-positive-integrations.json"]["integration_count"] == 55
    assert stats["gennova-positive-integrations.json"]["combination_effect_count"] == 29
    assert stats["negative-results.json"]["negative_result_count"] == 90
