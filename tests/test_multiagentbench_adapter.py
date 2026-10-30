"""Native configuration custody fixtures, not Agent/scoring validation."""

from copy import deepcopy
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
