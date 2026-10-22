"""CPU checks for category-level delivery bookkeeping, not model evidence."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/workload-data"


def test_category_checklist_preserves_every_class_and_source():
    catalogs = ROOT / "workload-preparation/data/catalogs"
    base = json.loads((catalogs / "workload_taxonomy_v1.json").read_text("utf-8"))
    refined = json.loads((catalogs / "workload_taxonomy_v2.json").read_text("utf-8"))
    replaced = {r["parent_family_id"] for r in refined["replacements"]}
    ids = {r["family_id"] for r in refined["replacements"]} | {
        r["family_id"]
        for r in base["canonical_families"]
        if r["family_id"] not in replaced
    }
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    gaps = json.loads((DOCS / "gap-sources-20261021.json").read_text("utf-8"))
    checklist = (DOCS / "WORKLOAD_EXECUTION_CHECKLIST.zh-CN.md").read_text("utf-8")
    assert len(ids) == 36
    assert all(checklist.count("· " + identifier + "\n") == 1 for identifier in ids)
    assert all(row["source_id"] in checklist for row in ledger["sources"])
    assert all("- [ ] " + row["id"] + "：" in checklist for row in gaps["sources"])


def test_incremental_receipt_does_not_claim_full_split():
    receipt = json.loads(
        (DOCS / "validation/gsm8k-incremental-20261022.json").read_text("utf-8")
    )
    assert receipt["selected_samples"] == [f"test:{i}" for i in range(3, 9)]
    assert receipt["verified_unique_tasks"] == 6
    assert receipt["scope"] == "INCREMENTAL_SAMPLE_VALIDATION_NOT_FULL_SUITE"
    assert "1319" in receipt["sample_count_caveat"]
