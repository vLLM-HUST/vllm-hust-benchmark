import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec


def test_original_article_is_not_rewritten_or_given_reference_summary():
    row = {
        "document": "  Original article.\nSecond paragraph.  ",
        "summary": " Separate reference. ",
        "id": "00042",
    }
    record = adapt_row(
        "XSUM", row, source={"source_id": "XSUM"}, spec=adapter_spec("XSUM"), index=0
    )
    assert record.task["input"] == {"document": row["document"]}
    assert record.task["task_id"] == row["id"]
    assert record.outcome_record() == {"summary": row["summary"]}
    assert record.replay["annotations"]["id"] == row["id"]
    assert record.replay["unmapped_field_names"] == []
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


@pytest.mark.parametrize(
    "field,value", [("document", []), ("summary", None), ("id", 42)]
)
def test_invalid_original_fields_are_not_silently_coerced(field, value):
    row = {"document": "Article", "summary": "Reference", "id": "42"}
    row[field] = value
    with pytest.raises(ValueError, match="original document|null required field"):
        adapt_row(
            "XSUM",
            row,
            source={"source_id": "XSUM"},
            spec=adapter_spec("XSUM"),
            index=0,
        )


def test_original_blank_document_is_preserved_not_filtered_or_filled():
    # Actual publisher parquet includes34 blank documents across the3 splits.
    row = {"document": "", "summary": "Original reference", "id": "42"}
    record = adapt_row(
        "XSUM", row, source={"source_id": "XSUM"}, spec=adapter_spec("XSUM"), index=0
    )
    assert record.task["input"] == {"document": ""}
    assert record.outcome_record() == {"summary": row["summary"]}
