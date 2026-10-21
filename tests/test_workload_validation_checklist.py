"""Keep dataset checkboxes aligned with source identities and actual evidence."""

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_checklist_covers_registered_sources_once():
    directory = ROOT / "docs" / "workload-data"
    ledger = json.loads(
        (directory / "dataset-validation-status.json").read_text(encoding="utf-8")
    )
    catalog = json.loads(
        (
            ROOT / "workload-preparation/data/catalogs/source_manifests_v1.json"
        ).read_text(encoding="utf-8")
    )
    rows = ledger["sources"]
    assert len(rows) == len({row["source_id"] for row in rows})
    assert {row["source_id"] for row in rows} == set(catalog["sources"])
    markdown = (directory / "DATASET_CHECKLIST.zh-CN.md").read_text(encoding="utf-8")
    table_rows = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in markdown.splitlines()
        if line.startswith("|")
    ]
    for row in rows:
        marker = "x" if row["checked"] else " "
        assert (
            sum(cells[:2] == [f"[{marker}]", row["source_id"]] for cells in table_rows)
            == 1
        )


def test_summary_counts_match_actual_checkboxes():
    directory = ROOT / "docs/workload-data"
    rows = json.loads(
        (directory / "dataset-validation-status.json").read_text(encoding="utf-8")
    )["sources"]
    table = [
        [cell.strip() for cell in line.split("|")[1:-1]]
        for line in (directory / "DATASET_CHECKLIST.zh-CN.md")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.startswith("|")
    ]
    for scope, title in (
        ("public", "公开候选来源"),
        ("enterprise_private", "企业私有来源"),
    ):
        selected = [row for row in rows if row["scope"] == scope]
        assert [
            title,
            str(len(selected)),
            str(sum(row["checked"] for row in selected)),
        ] in table


def test_checked_sources_require_execution_and_scoring_evidence():
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )
    for row in ledger["sources"]:
        if not row["checked"]:
            continue
        assert row["acquisition"] == "VERIFIED"
        assert row["read_adapter"] == "VERIFIED"
        assert row["server_execution"] == "COMPLETED"
        assert row["scoring"] in {"COMPLETED", "NON_SCORING_INTEGRITY_VERIFIED"}
        assert (
            len(set(row["selected_samples"]))
            >= ledger["completion_rule"]["minimum_preselected_samples"]
        )
        assert set(row["completed_samples"]) == set(row["selected_samples"])
        assert row["receipt"]
        receipt = (ROOT / row["receipt"]).resolve()
        assert receipt.is_relative_to(ROOT.resolve())
        assert receipt.is_file()
        evidence = json.loads(receipt.read_text(encoding="utf-8"))
        assert evidence["source_id"] == row["source_id"]
        assert evidence["status"] == "END_TO_END_EXECUTED_AND_SCORED"
        assert evidence["selected_samples"] == row["selected_samples"]
        assert evidence["server_inventory"]["backend"] == "vllm-ascend"


def test_arc_task_count_and_protocol_are_not_chat_or_full_suite_claims():
    evidence = json.loads(
        (
            ROOT / "docs/workload-data/validation/arc-challenge-npu-20261022.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["task_count"] == 3
    assert evidence["candidate_likelihood_requests"] == 12
    assert (
        evidence["model_answer_correct_count"]
        + evidence["model_answer_incorrect_count"]
        == 3
    )
    assert (
        evidence["protocol"]
        == "original multiple-choice loglikelihood; no chat conversion"
    )
    assert evidence["subsuite"] == "ARC-Challenge"
    assert (
        evidence["helper_sha256"]
        == hashlib.sha256(
            (ROOT / "scripts/run_arc_original_samples.py").read_bytes()
        ).hexdigest()
    )
