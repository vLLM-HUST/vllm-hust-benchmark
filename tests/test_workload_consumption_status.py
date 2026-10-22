"""Bookkeeping checks only; these tests do not rerun consumers or native tasks."""

import json
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


def test_locomo_final_input_consumption_does_not_hide_failed_native_execution():
    ledger = json.loads((DOCS / "dataset-validation-status.json").read_text("utf-8"))
    row = next(r for r in ledger["sources"] if r["source_id"] == "LOCOMO")
    final = json.loads(
        (ROOT / row["fixed_input_consumption_receipt"]).read_text("utf-8")
    )
    failure = json.loads((ROOT / row["execution_receipt"]).read_text("utf-8"))
    assert final["server_installed_original_question_fields_checked"] == 1986
    assert (
        final["server_prepare_resolve_roundtrips"]
        == final["consumer_records_per_mode"]
        == 3
    )
    assert final["consumer_modes_equal"] is True
    assert final["old_source_regression_records_per_mode"] == 9
    assert final["source_checkbox_eligible"] is row["checked"] is False
    assert final["native_task_executions_in_this_input_audit"] == 0
    assert failure["completed_model_generations"] == failure["scorer_runs"] == 0
    assert (
        row["server_execution"]
        == "FIRST_ORIGINAL_REQUEST_HTTP500_ENGINE_EXITED_NO_COMPLETION"
    )
