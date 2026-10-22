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
    assert (
        "653ad3518e5fcd78c958d86ea39bac6d8714fb3a" in guide
    )  # pragma: allowlist secret - public data Git commit
    assert (
        "4906910e9c5f9909d84e41d59f9f49dcb055a912" in guide
    )  # pragma: allowlist secret - public consumer Git commit
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
