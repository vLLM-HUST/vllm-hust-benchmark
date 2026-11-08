"""Invented xLAM field-contract controls, no publisher requests bundled."""

from copy import deepcopy
import json

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec


def test_original_tools_query_and_reference_are_separate():
    row = {
        "id": 7,
        "query": "invented fixture request",
        "tools": [{"name": "fixture_tool", "parameters": {"type": "object"}}],
        "answers": [{"name": "fixture_tool", "arguments": {"fixture": 1}}],
    }
    original = deepcopy(row)
    record = adapt_row(
        "XLAM_TOOL",
        row,
        source={"source_id": "XLAM_TOOL"},
        spec=adapter_spec("XLAM_TOOL"),
        index=0,
    )
    assert record.task["input"] == {k: row[k] for k in ("query", "tools")}
    assert record.outcome_record() == {"answers": row["answers"]}
    assert "answers" not in record.task["input"]
    assert record.replay["unmapped_field_names"] == []
    assert record.runtime["execution"] == "not_executed"
    assert record.task["task_id"] == "7"
    row["tools"].clear()
    assert record.task["input"]["tools"] == original["tools"]
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


@pytest.mark.parametrize("missing", ["id", "query", "tools", "answers"])
def test_missing_original_field_is_not_synthesized(missing):
    row = {"id": 0, "query": "fixture", "tools": [], "answers": []}
    del row[missing]
    with pytest.raises(ValueError):
        adapt_row(
            "XLAM_TOOL",
            row,
            source={"source_id": "XLAM_TOOL"},
            spec=adapter_spec("XLAM_TOOL"),
            index=0,
        )


def test_publisher_json_encoded_fields_remain_original_strings():
    row = {
        "id": 1,
        "query": "invented original-format fixture",
        "tools": json.dumps([{"name": "fixture", "parameters": {}}]),
        "answers": json.dumps([{"name": "fixture", "arguments": {}}]),
    }
    record = adapt_row(
        "XLAM_TOOL",
        row,
        source={"source_id": "XLAM_TOOL"},
        spec=adapter_spec("XLAM_TOOL"),
        index=0,
    )
    assert record.task["input"]["tools"] == row["tools"]
    assert record.outcome_record()["answers"] == row["answers"]
    assert record.replay["annotations"]["id"] == row["id"]
