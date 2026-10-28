"""Invented field fixtures only, not native benchmark examples or scores."""

from copy import deepcopy
import json

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def original_fixture():
    return {
        "task_id": 11,
        "text": " Original task.\n ",
        "code": "SCORER_ONLY_REFERENCE",
        "test_setup_code": "SCORER_ONLY_SETUP",
        "test_list": ["public_test_one", "public_test_two", "public_test_three"],
        "challenge_test_list": ["SCORER_ONLY_CHALLENGE"],
    }


def adapt(row):
    return adapt_row(
        "MBPP", row, source={"source_id": "MBPP"}, spec=adapter_spec("MBPP"), index=0
    )


def test_full_original_fields_and_reference_separation():
    row = original_fixture()
    record = adapt(row)
    assert record.task["task_id"] == "11"
    assert record.task["input"] == {key: row[key] for key in ("text", "test_list")}
    assert "SCORER_ONLY" not in json.dumps(record.input_record())
    assert record.outcome_record() == {
        key: row[key] for key in ("code", "test_setup_code", "challenge_test_list")
    }
    assert record.replay["annotations"] == {"task_id": 11}
    assert record.replay["unmapped_field_names"] == []
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()
    row["test_list"].append("mutated")
    assert len(record.task["input"]["test_list"]) == 3


@pytest.mark.parametrize(
    "key,value",
    [
        ("task_id", True),
        ("task_id", "11"),
        ("task_id", 0),
        ("code", []),
        ("test_setup_code", None),
        ("test_list", ["only_one"]),
        ("test_list", ["one", "two", 3]),
        ("challenge_test_list", "not_an_array"),
        ("challenge_test_list", [None]),
    ],
)
def test_invalid_fields_fail_closed(key, value):
    row = original_fixture()
    row[key] = value
    with pytest.raises(ValueError):
        adapt(row)


def test_sanitized_and_extra_fields_are_not_silently_dropped():
    row = original_fixture()
    sanitized = deepcopy(row)
    sanitized["prompt"] = sanitized.pop("text")
    for invalid in (sanitized, {**row, "unreviewed": "extra"}):
        with pytest.raises(ValueError):
            adapt(invalid)


def test_all_full_splits_registered_as_readable_data_not_native_completion():
    source = source_spec("MBPP")
    assert source["platform"] == "hf"
    revision = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"  # pragma: allowlist secret - public dataset revision
    assert source["revision"] == revision
    assert len(source["files"]) == 4
    assert sum(item["rows"] for item in source["files"]) == 974
    assert all(
        item["role"] == "data" and item["path"].startswith("full/")
        for item in source["files"]
    )
    assert source.get("native_execution_complete") is not True
