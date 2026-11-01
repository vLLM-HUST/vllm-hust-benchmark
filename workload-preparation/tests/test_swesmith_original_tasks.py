"""Inert fixtures prove separation, not original Agent execution."""

from copy import deepcopy
import json

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec
from vllm_hust_workload_data.swesmith import restore_original_task


def fixture_row():
    return {
        "instance_id": "fixture__repo.fixture",
        "patch": "fixture bug patch",
        "FAIL_TO_PASS": ["fixture.test"],
        "PASS_TO_PASS": [],
        "image_name": "fixture/image",
        "repo": "fixture/repo",
        "problem_statement": "",
    }


def adapt(row):
    return adapt_row(
        "SWESMITH",
        row,
        source={"source_id": "SWESMITH"},
        spec=adapter_spec("SWESMITH"),
        index=2,
    )


def test_empty_original_issue_and_bug_patch_roundtrip_without_leak():
    row = fixture_row()
    record = adapt(row)
    assert restore_original_task(record) == row
    assert record.task["input"]["problem_statement"] == ""
    assert "fixture bug patch" not in json.dumps(record.input_record())
    assert "fixture.test" not in json.dumps(record.input_record())
    assert record.runtime["execution"] == "not_executed"
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()
    restored = restore_original_task(record)
    restored["FAIL_TO_PASS"].append("new")
    assert restore_original_task(record) == row


@pytest.mark.parametrize(
    "field,value",
    [
        ("problem_statement", None),
        ("FAIL_TO_PASS", "[]"),
        ("PASS_TO_PASS", [1]),
        ("image_name", ""),
    ],
)
def test_invalid_original_fields_rejected(field, value):
    row = deepcopy(fixture_row())
    row[field] = value
    with pytest.raises(ValueError, match="seven-field"):
        adapt(row)


def test_unknown_fields_not_silently_lost():
    row = fixture_row() | {"unreviewed": "field"}
    with pytest.raises(ValueError, match="seven-field"):
        adapt(row)
