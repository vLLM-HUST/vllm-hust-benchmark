"""Inert field-contract fixtures; not official data or native execution."""

from copy import deepcopy

import pytest

from vllm_hust_workload_data.livecodebench import adapt_case
from vllm_hust_workload_data.registry import adapter_spec


@pytest.mark.parametrize(
    "scenario", ["codegeneration", "codeexecution", "testoutputprediction"]
)
def test_original_fields_reconstruct_without_answer_input_leak(scenario):
    row = {"question_id": "fixture-id", "contest_date": "inert-date"}
    if scenario == "codegeneration":
        row.update(
            question_title="inert title",
            question_content="inert task",
            starter_code="",
            public_test_cases="opaque-public",
            private_test_cases="opaque-private",
            metadata="opaque-metadata",
        )
    elif scenario == "codeexecution":
        row.update(
            question_id=7,
            id="fixture-case",
            function_name="inert",
            code="not executed",
            input="opaque-input",
            output="opaque-answer",
            numsteps=1,
        )
    else:
        row.update(
            question_title="inert title",
            question_content="inert task",
            starter_code="",
            function_name="inert",
            test_id=0,
            test="opaque-test",
        )
    original = deepcopy(row)
    record = adapt_case(row, source={"source_id": "LIVECODEBENCH"}, index=0)
    assert record.task["native_unit"] == scenario
    rebuilt = {
        **record.task["input"],
        **record.replay["original_annotations"],
        **record.outcome,
    }
    assert rebuilt == original == row
    assert not set(record.task["input"]) & set(record.outcome)
    assert record.replay["synthetic_fallback"] is False
    with pytest.raises(ValueError, match="not a ready"):
        record.request_body()


def test_registry_exposes_original_adapter_not_official_execution_claim():
    spec = adapter_spec("LIVECODEBENCH")
    assert spec["mapping"] == "livecodebench_original_tasks"
    assert "PENDING" in spec["status"]


def test_missing_native_fields_are_rejected():
    with pytest.raises(ValueError, match="incomplete"):
        adapt_case({"question_id": "fixture", "test_id": 0}, source={}, index=0)
