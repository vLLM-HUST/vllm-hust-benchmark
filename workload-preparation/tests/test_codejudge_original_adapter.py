"""Original published-prompt mapping fixtures, not source data or model evidence."""

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def original_fixture():
    return {
        "data_id": 1,
        "task_id": 77,
        "source": "fixture-generator",
        "url": "https://example.invalid/fixture-problem",
        "statement": "fixture original problem",
        "code": "fixture original code",
        "input": "published fixture prompt with unchanged one-shot example",
        "answer": "C",
    }


def test_original_full_prompt_reference_and_grouping_preserved():
    row = original_fixture()
    record = adapt_row(
        "CODEJUDGE_EVAL",
        row,
        source={"source_id": "CODEJUDGE_EVAL"},
        spec=adapter_spec("CODEJUDGE_EVAL"),
        index=0,
    )
    assert record.task["input"] == {"input": row["input"]}
    assert record.outcome == {"answer": "C"}
    assert record.replay["annotations"]["task_id"] == 77
    assert record.replay["annotations"]["data_id"] == 1
    assert record.task["input"] | record.outcome | record.replay["annotations"] == row
    assert record.replay["unmapped_field_names"] == []
    assert record.replay["synthetic_fallback"] is False
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


@pytest.mark.parametrize(
    "key,value", [("input", ""), ("answer", "unknown"), ("data_id", True)]
)
def test_invalid_original_fields_fail_not_generated(key, value):
    row = original_fixture()
    row[key] = value
    with pytest.raises(ValueError):
        adapt_row(
            "CODEJUDGE_EVAL",
            row,
            source={"source_id": "CODEJUDGE_EVAL"},
            spec=adapter_spec("CODEJUDGE_EVAL"),
            index=0,
        )


def test_six_configuration_files_not_six_new_datasets():
    manifest = source_spec("CODEJUDGE_EVAL")
    assert len(manifest["files"]) == 6
    assert sum(file["rows"] for file in manifest["files"]) == 11160
    assert len({file["sha256"] for file in manifest["files"]}) == 6
    assert "run_codejudge_original.py" in manifest["native_reference_command"]
    assert "not author regex" in manifest["native_reference_command"]
