"""CPU checks for category-level delivery bookkeeping, not model evidence."""

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/workload-data"


def test_target_repository_inventory_preserves_existing_versions_and_aliases():
    plan = (DOCS / "WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md").read_text("utf-8")
    section = plan.split("### 8.6 ", 1)[1].split("## 九、", 1)[0]
    matrix = section.split("以下表只说明", 1)[1].split("#### 8.6.2", 1)[0]
    rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in matrix.splitlines()
        if line.startswith("|") and "`" in line
    ]
    assert len(rows) == len({row[0] for row in rows}) == 31
    assert sum(len(re.findall(r"`([^`]+)`", row[1])) for row in rows) == 35
    assert sum(row[0].startswith("PAIO-") for row in rows) == 8
    assert "SWE_BENCH" not in {row[0] for row in rows}
    assert (
        # pragma: allowlist nextline secret - public source revision
        "6e0cc61db57487778b2147cc18bf5a726cac4817" in section
    )
    for path in (
        "src/vllm_hust_benchmark/mmlu_pro_pyramidkv.py",
        "scripts/audit_pyramidkv_hle.py",
        "src/vllm_hust_benchmark/swe_bench_pro.py",
        "src/vllm_hust_benchmark/frontierscience_inputs.py",
        "scripts/prepare_pyramidkv_terminal.py",
    ):
        assert path in section
    assert "不是整个benchmark仓的全部已有成果" in section
    assert "缺backend、缺评分器、坏协议或服务崩溃" in section


def test_inventory_snapshot_separates_read_run_consume_and_legacy_checks():
    """Frozen PR inventory, not current model execution or whole-suite quality."""
    plan = (DOCS / "WORKLOAD_PR_REORGANIZATION_PLAN.zh-CN.md").read_text("utf-8")
    section = plan.split("#### 8.5.2 ", 1)[1].split("#### 8.5.3 ", 1)[0]
    rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in section.splitlines()
        if line.startswith("|") and "[x]" in line
    ]
    assert len(rows) == len({row[0] for row in rows}) == 37
    assert all(row[1].startswith("[x]") for row in rows)
    assert sum(row[2].startswith("[x]") for row in rows) == 28
    assert (
        sum(
            row[4].startswith("[x]")
            and "本地CPU-only消费回执" not in row[4]
            and "本地补充消费回执" not in row[4]
            for row in rows
        )
        == 23
    )
    assert (
        sum(
            "本地补充消费回执" in row[4] or "本地CPU-only消费回执" in row[4]
            for row in rows
        )
        == 14
    )
    assert sum(row[4].startswith("[x]") for row in rows) == 37
    assert sum("旧[x]" in row[5] for row in rows) == 19
    assert (
        sum(
            "已执行并得到评分/validator结果" in row[3] or "固定harness评分0/3" in row[3]
            for row in rows
        )
        == 13
    )
    assert sum("响应完整性；无语义gold" in row[3] for row in rows) == 11
    assert "MATH500" not in {row[0] for row in rows}
    initial = json.loads(
        (DOCS / "validation/natural-instructions-original-npu-20261023.json").read_text(
            "utf-8"
        )
    )
    extra = json.loads(
        (
            DOCS / "validation/ni-editing-extraction-original-npu-20261023.json"
        ).read_text("utf-8")
    )
    tasks = initial["task_results"] + extra["task_results"]
    assert len({task["task_file"] for task in tasks}) == len(tasks) == 5
    assert sum(task["executed_unique"] for task in tasks) == 361


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
    assert len(ids) == 38
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
