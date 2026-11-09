from copy import deepcopy

import pytest

from vllm_hust_workload_data.stabletoolbench import adapt_case, restore_original_task


def original():
    return {
        "query_id": 588,
        "query": "fixture query",
        "api_list": [{"api_name": "first"}, {"api_name": "second"}],
        "relevant APIs": [["tool", "first"]],
        "publisher_extension": {"keep": True},
    }


def test_native_task_roundtrip_keeps_tools_and_separates_reference():
    row = original()
    record = adapt_case(row, source={"source_id": "STABLETOOLBENCH"}, index=0)
    assert restore_original_task(record) == row
    assert "relevant APIs" not in record.input_record()["task"]["input"]
    assert len(record.task["input"]["api_list"]) == 2
    assert record.runtime["kind"] == "native_environment"
    record.task["input"]["api_list"][0]["api_name"] = "changed"
    assert row == original()


@pytest.mark.parametrize("field", ["query_id", "query", "api_list", "relevant APIs"])
def test_missing_original_fields_are_not_fabricated(field):
    row = original()
    del row[field]
    with pytest.raises(ValueError, match="original StableToolBench"):
        adapt_case(row, source={"source_id": "STABLETOOLBENCH"}, index=0)


def test_restore_rejects_identity_change_and_reference_leak():
    record = adapt_case(original(), source={"source_id": "STABLETOOLBENCH"}, index=0)
    changed = deepcopy(record)
    changed.task["task_id"] = "different"
    with pytest.raises(ValueError, match="identity changed"):
        restore_original_task(changed)
    record.task["input"]["relevant APIs"] = []
    with pytest.raises(ValueError, match="separation"):
        restore_original_task(record)
