"""Keep dataset checkboxes aligned with source identities and actual evidence."""

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_partial_history_execution_and_retrieval_preparation_stay_distinct():
    directory = ROOT / "docs/workload-data"
    sources = json.loads(
        (directory / "dataset-validation-status.json").read_text(encoding="utf-8")
    )["sources"]
    rows = {row["source_id"]: row for row in sources}
    history = rows["OASST1"]
    receipt = json.loads((ROOT / history["receipt"]).read_text(encoding="utf-8"))
    assert history["selected_samples"] == history["completed_samples"] == [23, 28, 68]
    assert history["selected_samples"] == [
        row["row_index"] for row in receipt["samples"]
    ]
    assert history["server_execution"] == "COMPLETED_ORIGINAL_HISTORY_CONTINUATIONS"
    assert all(row["finish_reason"] == "stop" for row in receipt["samples"])
    assert receipt["quality_scored"] is False
    assert receipt["live_episode_validated"] is False
    assert history["checked"] is False
    retrieval = rows["QRECC"]
    assert retrieval["acquisition"] == "VERIFIED_ORIGINAL_TEST_ARCHIVE_AND_MEMBER"
    assert retrieval["read_adapter"] == "VERIFIED_INSTALLED_CLI_3_SAMPLES"
    assert retrieval["server_execution"] == retrieval["scoring"] == "NOT_RUN"
    assert retrieval["completed_samples"] == []
    assert retrieval["checked"] is False


def test_humaneval_generation_without_functional_scoring_stays_unchecked():
    directory = ROOT / "docs/workload-data"
    ledger = json.loads(
        (directory / "dataset-validation-status.json").read_text(encoding="utf-8")
    )
    row = next(r for r in ledger["sources"] if r["source_id"] == "HUMANEVAL")
    receipt = json.loads((ROOT / row["receipt"]).read_text(encoding="utf-8"))
    assert row["checked"] is False
    assert row["completed_samples"] == []
    assert row["scoring"] == "BLOCKED_NO_VERIFIED_SANDBOX"
    assert receipt["complete_task_validation"] is False
    assert receipt["source_checkbox_eligible"] is False
    assert receipt["sandbox_probe"]["generated_code_executed"] is False
    assert len(receipt["samples"]) == 3
    assert (
        receipt["helper_sha256"]
        == hashlib.sha256(
            (ROOT / "scripts/run_humaneval_original_generation.py").read_bytes()
        ).hexdigest()
    )


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
        if row["scoring"] == "COMPLETED":
            assert evidence["status"] == "END_TO_END_EXECUTED_AND_SCORED"
        else:
            assert evidence["status"] in {
                "REQUESTS_EXECUTED_AND_INTEGRITY_VALIDATED_NO_QUALITY_ORACLE",
                "ORIGINAL_FIELDS_GENERATED_AND_INTEGRITY_VERIFIED_NO_QUALITY_SCORE",
                "ORIGINAL_MESSAGES_GENERATED_AND_INTEGRITY_VERIFIED_NO_QUALITY_SCORE",
            }
            assert evidence["quality_claim"] is False
            assert evidence["scoring"] == "NON_SCORING_INTEGRITY_VERIFIED"
            assert len(evidence["samples"]) == 3
            assert [s["record_id"] for s in evidence["samples"]] == evidence[
                "selected_samples"
            ]
            assert all(
                s["response_validation"]["status"] == "RESPONSE_INTEGRITY_VERIFIED"
                for s in evidence["samples"]
            )
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
    # Historical receipt pins the helper that ran ARC-Challenge before the
    # shared ARC-Easy support was added.
    assert (
        evidence["helper_sha256"]
        == "f45dbe69e17309fcd74dd49fc3b74ce8dc6d7921a4c46c3268c1dcc593088568"
    )


def test_enterprise_checks_are_not_quality_scores_and_preserve_full_inputs():
    directory = ROOT / "docs/workload-data/validation"
    records = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in directory.glob("paio-*-npu-20261022.json")
    ]
    assert len(records) == 8
    assert sum(r["sample_count"] for r in records) == 24
    helper = hashlib.sha256(
        (ROOT / "scripts/run_source_request_samples.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    ).hexdigest()
    checker = hashlib.sha256(
        (ROOT / "src/vllm_hust_benchmark/workload_response_validation.py")
        .read_bytes()
        .replace(b"\r\n", b"\n")
    ).hexdigest()
    for record in records:
        assert record["quality_claim"] is False
        assert record["production_trace_representativity"] == "NOT_ESTABLISHED"
        assert record["source_asset"]["matches_catalog_file_sha256"] is True
        assert record["software_provenance"]["model_execution_helper_sha256"] == helper
        assert record["software_provenance"]["response_validator_sha256"] == checker
        for sample in record["samples"]:
            assert sample["finish_reasons"] == ["stop"]
            assert sample["full_prompt_token_check"] == "MATCHED_SERVER_USAGE"
            assert sample["server_prompt_tokens"] == sample["preflight_prompt_tokens"]


def test_model_artifact_records_exact_publisher_weights_not_just_a_name():
    artifact = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/qwen35-27b-deployable-artifact-20261022.json"
        ).read_text(encoding="utf-8")
    )
    assert artifact["publisher_all_shards_verified"] is True
    assert len(artifact["files"]) == 11
    assert all(row["sha256"] == row["publisher_sha256"] for row in artifact["files"])
    assert artifact["publisher_runtime_files_verified"] is True
    assert all(row["publisher_matches"] for row in artifact["runtime_files"])
