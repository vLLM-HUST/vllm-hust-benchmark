import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec


def test_original_bill_input_excludes_title_and_reference():
    row = {
        "text": "  Original bill.\nArticle2. ",
        "summary": "Gold",
        "title": "Bill title",
    }
    record = adapt_row(
        "BILLSUM",
        row,
        source={"source_id": "BILLSUM", "file_sha256": "fixture-sha"},
        spec=adapter_spec("BILLSUM"),
        index=0,
    )
    assert record.task["input"] == {"text": row["text"]}
    assert record.outcome_record() == {"summary": "Gold"}
    assert record.replay["annotations"]["title"] == "Bill title"
    assert record.replay["unmapped_field_names"] == []
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


@pytest.mark.parametrize("field", ["text", "summary", "title"])
def test_no_silent_coercion_of_original_fields(field):
    row = {"text": "Bill", "summary": "Gold", "title": "Title"}
    row[field] = ["not a string"]
    with pytest.raises(ValueError, match="original text|invalid text field"):
        adapt_row(
            "BILLSUM",
            row,
            source={"source_id": "BILLSUM"},
            spec=adapter_spec("BILLSUM"),
            index=0,
        )
