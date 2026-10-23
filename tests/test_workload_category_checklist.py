"""CPU checks for category-level delivery bookkeeping, not model evidence."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/workload-data"


def test_forty_class_plan_and_migration_audit_do_not_upgrade_runtime_evidence():
    plan = (DOCS / "WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md").read_text("utf-8")
    ids = re.findall(r"^\| W(\d{2}) ", plan, flags=re.MULTILINE)
    assert ids == [f"{index:02d}" for index in range(1, 41)]
    anchors = re.findall(r'<a id="(src-[^"]+)"></a>', plan)
    assert len(anchors) == len(set(anchors)) == 110
    assert all(
        source in plan
        for source in (
            "MMLU_PRO",
            "HLE_VERIFIED",
            "SWE_BENCH_PRO",
            "FRONTIERSCIENCE",
            "TERMINAL_BENCH_2_1",
        )
    )
    for source in json.loads(
        (DOCS / "dataset-validation-status.json").read_text("utf-8")
    )["sources"]:
        assert source["source_id"] in plan
    audit = json.loads(
        (
            DOCS / "validation/migration-existing-capability-audit-20261023.json"
        ).read_text("utf-8")
    )
    assert audit["status"] == "PASS_SOFTWARE_FIXTURES_NOT_REAL_TASKS"
    assert audit["baseline_adapter_definitions"] == 45
    assert audit["compared_data_adapters"] == len(audit["rows"]) == 42
    assert len(audit["intentionally_excluded_qbi_definitions"]) == 3
    assert (
        audit["model_calls"]
        == audit["native_tasks_run"]
        == audit["new_datasets_acquired"]
        == 0
    )
    assert audit["checks"]["baseline_adapter_ids_missing"] == []
    assert audit["checks"]["baseline_fixed_data_file_identities_covered"] == 38


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


def test_resumed_gsm_coverage_counts_actual_ids_not_upstream_effective_total():
    receipt = json.loads(
        (
            ROOT / "docs/workload-data/validation/gsm8k-resumed-coverage-20261022.json"
        ).read_text("utf-8")
    )
    assert receipt["unique_verified_tasks"] == 201 < receipt["original_test_size"]
    assert receipt["original_doc_id_range_inclusive"] == [0, 200]
    assert receipt["selection_matches_actual_logged_doc_ids"] is True
    assert (
        receipt["prior_frozen_snapshot_unique_tasks"]
        + receipt["additional_pre_restart_unique_tasks_recovered"]
        + receipt["new_post_restart_unique_tasks"]
        == 201
    )
    assert receipt["new_batch"]["selected_doc_ids"] == list(range(195, 201))
    assert receipt["new_batch"]["verified_unique_tasks"] == 6
    assert receipt["new_batch"]["upstream_reported_effective_count"] == 1319
    assert receipt["full_suite_claim"] is False
