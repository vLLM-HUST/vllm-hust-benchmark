"""Native configuration custody fixtures, not Agent/scoring validation."""

from copy import deepcopy
import json
from pathlib import Path
import pytest

from vllm_hust_workload_data.multiagentbench import (
    adapt_case,
    restore_original_configuration,
)


@pytest.mark.parametrize("scenario", ["database", "minecraft", "research"])
def test_original_configuration_roundtrip_and_reference_separation(scenario):
    row = {
        "task": {"content": "fixture task"},
        "agents": [{"private_fixture": "agent-only"}],
        "relationships": [["a", "b"]],
        "environment": {"task_id": 7},
        "unknown_extension": {"preserve": [1, 2]},
    }
    if scenario == "database":
        row["task"].update(
            labels=["reference-only"],
            root_causes=["cause-only"],
            number_of_labels_pred=1,
        )
    source = {
        "source_id": "MULTIAGENTBENCH",
        "file": f"multiagentbench/{scenario}/{scenario}_main.jsonl",
        "file_sha256": "fixture-digest",
    }
    original = deepcopy(row)
    record = adapt_case(row, source=source, index=0)
    assert restore_original_configuration(record) == original == row
    assert "labels" not in record.task["input"]
    assert "labels" not in record.runtime["original_configuration"]["task"]
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()
    record.runtime["original_configuration"]["unknown_extension"]["preserve"].append(3)
    assert row == original


def test_input_reference_tampering_rejected():
    source = {
        "source_id": "MULTIAGENTBENCH",
        "file": "multiagentbench/database/database_main.jsonl",
        "file_sha256": "fixture",
    }
    record = adapt_case(
        {
            "task": {"content": "fixture", "labels": ["reference"]},
            "agents": [],
            "relationships": [],
            "environment": {},
        },
        source=source,
        index=0,
    )
    record.task["input"]["labels"] = ["leaked"]
    with pytest.raises(ValueError, match="separation"):
        restore_original_configuration(record)


def test_source_handoff_evidence_does_not_claim_native_completion():
    root = Path(__file__).resolve().parents[1]
    ledger = json.loads(
        (root / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )
    row = ledger["delivery_inventory"]["MULTIAGENTBENCH"]
    handoff = row["research_source_checkout_handoff"]
    assert handoff["original_index_records"] == 500
    assert handoff["public_and_research_full_record_roundtrips"] == 500
    assert handoff["restored_original_configurations_equal"] == 500
    assert handoff["research_code_or_default_dependency_changed"] is False
    assert handoff["actual_ci_artifact_verified"] is False
    assert handoff["model_calls"] == 0
    assert handoff["native_execution"] is False and handoff["official_scoring"] is False
    assert handoff["source_checkbox_eligible"] is False and row["checked"] is False


def test_actual_ci_handoff_still_does_not_claim_native_completion():
    root = Path(__file__).resolve().parents[1]
    row = json.loads(
        (root / "docs/workload-data/dataset-validation-status.json").read_text("utf-8")
    )["delivery_inventory"]["MULTIAGENTBENCH"]
    handoff = row["actual_ci_artifact_handoff"]
    assert handoff["actual_ci_run"] == 41039829015
    assert handoff["actual_ci_artifact"] == 12970235878
    assert handoff["installed_runtime_payloads"] == 72
    assert handoff["public_and_unchanged_research_roundtrips"] == 500
    assert handoff["index_rebuilt_or_changed"] is False
    assert handoff["research_default_dependency_changed"] is False
    assert handoff["model_calls"] == 0
    assert handoff["native_execution"] is False and handoff["official_scoring"] is False
    assert handoff["source_checkbox_eligible"] is False and row["checked"] is False
