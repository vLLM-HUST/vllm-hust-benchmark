import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_closeout_keeps_history_and_explicit_reopening_scope():
    data = json.loads((ROOT / "docs/issue326-retained-history.json").read_text())
    assert data["schema"] == "issue326-active-consumer-closeout/v1"
    assert data["central_drift"] == []
    assert data["organization_repositories_scanned"] >= len(data["retained_history"])
    assert len(data["reopen_only_for"]) == 4
    for row in data["retained_history"]:
        assert row["disposition"] == "retain-reviewed-baseline-not-zero-consumer-proof"
        assert row["paths"] or row["assets"]
    assert {row["id"] for row in data["missing_and_external"]} == {
        "old-33-probes",
        "profiler-m0-validation-and-g4",
        "profiler-g1-g4-and-temporary-trace",
        "old-paper-pdf-digest",
        "private-issue19-and-calibration",
        "hardware-and-review",
    }
    # This inventory is a point-in-time receipt, not the mutable live index.
    # Refreshing website navigation must not rewrite its historical identity.
    receipt = (ROOT / "docs/issue326-closeout.md").read_text()
    assert data["website_commit"] in receipt
    assert data["publication_commit"] in receipt


def test_current_navigation_capture_has_no_production_error():
    sources = json.loads((ROOT / "docs/data-entrypoints-sources.json").read_text())
    assert sources["production_manifest_error"] is None
