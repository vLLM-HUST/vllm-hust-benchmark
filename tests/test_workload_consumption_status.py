"""Bookkeeping checks only; these tests do not rerun consumers or native tasks."""

import json
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/workload-data"


def test_fixed_consumption_receipts_are_linked_without_erasing_native_gaps():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    rows = {row["source_id"]: row for row in ledger["sources"]}
    for source in (
        "AZURE_LLM",
        "BURSTGPT",
        "JSONSCHEMABENCH",
        "MOONCAKE",
        "TOOLACE",
        "TRACELAB",
    ):
        assert "consumer pending" not in rows[source]["blocker"]
        assert "research-consumer validation pending" not in rows[source]["blocker"]
        assert rows[source]["checked"] is False
    guide = (DOCS / "DATA_CONSUMPTION_VALIDATION.zh-CN.md").read_text("utf-8")
    data_commit = "653ad3518e5fcd78c958d86ea39bac6d8714fb3a"  # pragma: allowlist secret - public Git commit
    consumer_commit = "4906910e9c5f9909d84e41d59f9f49dcb055a912"  # pragma: allowlist secret - public Git commit
    assert data_commit in guide
    assert consumer_commit in guide
    assert "language-model-only" in guide
    assert "功能评分仍未执行" in guide
    assert "不因此打勾" in guide


def test_paws_acquisition_failure_is_not_adapter_or_task_validation():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = next(r for r in ledger["sources"] if r["source_id"] == "PAWS")
    receipt = json.loads((ROOT / row["receipt"]).read_text("utf-8"))
    assert receipt["data_obtained"] is False
    assert receipt["reader_validated"] is False
    assert receipt["source_checkbox_eligible"] is False
    assert (
        receipt["model_calls"]
        == receipt["native_task_executions"]
        == receipt["scorer_runs"]
        == 0
    )
    assert set(receipt["publisher_asset_attempts"].values()) == {"HTTP_403"}
    assert set(receipt["fixed_hf_metadata_attempts"].values()) == {"CONNECTION_TIMEOUT"}
    assert row["server_execution"] == row["scoring"] == "NOT_RUN"
    assert row["checked"] is False


def test_locomo_input_audit_and_prior_failure_remain_distinct_from_sampled_completion():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = next(r for r in ledger["sources"] if r["source_id"] == "LOCOMO")
    final = json.loads(
        (ROOT / row["fixed_input_consumption_receipt"]).read_text("utf-8")
    )
    failure = json.loads((ROOT / row["prior_failure_receipt"]).read_text("utf-8"))
    sampled = json.loads((ROOT / row["execution_receipt"]).read_text("utf-8"))
    assert final["server_installed_original_question_fields_checked"] == 1986
    assert (
        final["server_prepare_resolve_roundtrips"]
        == final["consumer_records_per_mode"]
        == 3
    )
    assert final["consumer_modes_equal"] is True
    assert final["old_source_regression_records_per_mode"] == 9
    assert final["source_checkbox_eligible"] is False
    assert final["native_task_executions_in_this_input_audit"] == 0
    assert failure["completed_model_generations"] == failure["scorer_runs"] == 0
    assert failure["model_requests_attempted"] == 1
    assert row["checked"] is True
    assert row["server_execution"] == row["scoring"] == "COMPLETED"
    assert sampled["status"] == "END_TO_END_EXECUTED_AND_SCORED"
    assert sampled["selected_samples"] == list(range(6))
    assert sampled["aggregate"]["publisher_f1_mean"] == 0.5337301587301587
    assert row["execution_attempt_receipts"] == [
        "docs/workload-data/validation/locomo-original-qa-recovery-20261023.json",
        "docs/workload-data/validation/locomo-original-qa-recovery-batch2-20261023.json",
    ]


def test_gsm8k_tail_receipt_tracks_three_actual_ids_not_harness_effective_count():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = next(r for r in ledger["sources"] if r["source_id"] == "GSM8K")
    receipt = json.loads(
        (ROOT / row["supplemental_execution_receipts"][0]).read_text("utf-8")
    )
    assert receipt["status"] == "END_TO_END_EXECUTED_AND_SCORED"
    assert receipt["selected_samples"] == ["test:1316", "test:1317", "test:1318"]
    assert receipt["verified_unique_tasks"] == 3
    metrics = receipt["metrics"]["gsm8k"]
    assert metrics["exact_match,strict-match"] == 1.0
    assert metrics["exact_match,flexible-extract"] == 1.0
    assert receipt["sample_counts"]["gsm8k"] == {"original": 1319, "effective": 1319}
    assert "counts three tasks only" in receipt["harness_cardinality_caveat"]
    assert receipt["prior_failed_invocation"]["model_generations"] == 0
    assert receipt["raw_requests_answers_and_model_outputs_published"] is False
    assert row["supplemental_execution_receipts"] == [
        "docs/workload-data/validation/gsm8k-tail-npu-20261023.json"
    ]


def test_gsm8k_cumulative_coverage_includes_prior_audit_and_tail_without_pooled_score():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = next(r for r in ledger["sources"] if r["source_id"] == "GSM8K")
    receipt = json.loads((ROOT / row["cumulative_coverage_receipt"]).read_text("utf-8"))
    assert row["checked"] is True
    expected_revision = "740312add88f781978c0658806c59bc2815b9866"  # pragma: allowlist secret - public dataset revision
    assert receipt["source_revision"] == expected_revision
    assert receipt["evidence"][0]["unique_task_count"] == 201
    assert receipt["evidence"][1]["selected_samples"] == [
        "test:1316",
        "test:1317",
        "test:1318",
    ]
    assert receipt["evidence"][1]["disjoint_from_prior_range"] is True
    assert receipt["cumulative"]["unique_task_count"] == 204
    assert receipt["cumulative"]["coverage_fraction"] == "204/1319"
    assert receipt["cumulative"]["pooled_accuracy_reported"] is False
    assert receipt["raw_tasks_answers_and_model_outputs_published"] is False


def test_mmlu_pro_chemistry_receipt_is_supplemental_and_sample_scoped():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = next(r for r in ledger["sources"] if r["source_id"] == "MMLU_PRO")
    receipt_path = row["supplemental_execution_receipts"][0]
    receipt = json.loads((ROOT / receipt_path).read_text("utf-8"))
    assert row["checked"] is True
    assert row["supplemental_samples"] == [
        "chemistry:0",
        "chemistry:1",
        "chemistry:2",
    ]
    assert receipt["status"] == "END_TO_END_EXECUTED_AND_SCORED"
    assert receipt["selected_samples"] == row["supplemental_samples"]
    assert receipt["task_counts"] == {
        "chemistry_original": 1132,
        "executed_unique": 3,
        "entire_test_split": 12032,
    }
    assert receipt["metrics"]["mmlu_pro_chemistry"]["exact_match,custom-extract"] == 1.0
    assert receipt["raw_questions_answers_and_model_outputs_published"] is False
    assert "其余12个学科及全量未跑" in row["blocker"]


def test_ai2_arc_easy_receipt_is_distinct_subset_and_sample_scoped():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = next(r for r in ledger["sources"] if r["source_id"] == "AI2_ARC")
    receipt = json.loads(
        (ROOT / "docs/workload-data/validation/arc-easy-npu-20261023.json").read_text(
            "utf-8"
        )
    )
    assert row["checked"] is True
    assert row["supplemental_samples"] == [
        "ARC-Easy/test:0",
        "ARC-Easy/test:1",
        "ARC-Easy/test:2",
    ]
    assert receipt["status"] == "END_TO_END_EXECUTED_AND_SCORED"
    assert receipt["selected_samples"] == ["test:0", "test:1", "test:2"]
    assert receipt["sample_counts"] == {"original": 2376, "effective": 3}
    # Preserve the historical raw remote file hash (uploaded with CRLF).
    assert (
        receipt["helper_sha256"]
        == (
            "8f9097547fb756251a5cddae17d9d246968db4e8c76e25ac4ac0cad051ef2161"  # pragma: allowlist secret - public runner file SHA256
        )
    )
    # The historical receipt pins the script actually executed then. The
    # range-capable current runner is independently executed and pinned by
    # the new tail audit; never rewrite the old execution identity.
    helper = (ROOT / "scripts/run_arc_original_samples.py").read_bytes()
    current = json.loads(
        (DOCS / "validation/arc-tail-native-audit-20261024.json").read_text("utf-8")
    )
    assert (
        hashlib.sha256(helper.replace(b"\r\n", b"\n")).hexdigest()
        == next(item for item in current["checks"] if item["suite"] == "ARC-Easy")[
            "runner_sha256"
        ]
    )
    assert receipt["metrics"]["acc,none"] == 1 / 3
    assert receipt["model_answer_correct_count"] == 1
    assert receipt["model_answer_incorrect_count"] == 2
    assert receipt["raw_tasks_answers_and_model_outputs_published"] is False
