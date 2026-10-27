"""Artificial software fixtures only; not publisher data or native execution."""

from copy import deepcopy

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.docvqa import verify_question
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def fixture_row():
    return {
        "questionId": "fixture-question",
        "question": "fixture question",
        "question_types": ["fixture type"],
        "image": {
            "bytes": b"\x89PNG\r\n\x1a\nfixture",
            "path": "documents/fixture.png",
        },
        "docId": 7,
        "ucsf_document_id": "fixture-document",
        "ucsf_document_page_no": "001",
        "answers": ["fixture reference", "fixture alternate"],
        "data_split": "val",
    }


def test_original_fields_answers_and_media_preserved_without_prompt_rendering():
    original = fixture_row()
    record = adapt_row(
        "DOCVQA",
        original,
        source={"source_id": "DOCVQA", "file": "fixture.parquet"},
        spec=adapter_spec("DOCVQA"),
        index=0,
    )
    assert record.task["task_id"] == original["questionId"]
    assert record.task["input"] == {
        "question": original["question"],
        "image": original["image"],
    }
    assert record.outcome_record() == {"answers": original["answers"]}
    assert {
        **record.task["input"],
        **record.outcome_record(),
        **record.replay["annotations"],
    } == original
    assert record.replay["unmapped_field_names"] == []
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()
    record.task["input"]["image"]["path"] = "changed-fixture"
    assert original == fixture_row()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("docId", True),
        ("docId", "7"),
        ("questionId", 1),
        ("answers", "reference"),
        ("answers", [1]),
        ("question_types", None),
        ("image", {"bytes": b"not png", "path": "x.png"}),
        ("image", {"bytes": b"\x89PNG\r\n\x1a\n", "path": "../x.png"}),
        ("image", {"bytes": b"\x89PNG\r\n\x1a\n", "path": "/x.png"}),
        ("image", {"bytes": b"\x89PNG\r\n\x1a\n", "path": "C:/x.png"}),
    ],
)
def test_no_type_coercion_or_unsafe_media_paths(field, value):
    original = fixture_row()
    original[field] = deepcopy(value)
    with pytest.raises(ValueError):
        verify_question(original)


def test_pinned_original_validation_inventory_not_delivery_claim():
    source = source_spec("DOCVQA")
    assert source["platform"] == "hf"
    assert len(source["files"]) == 6
    assert sum(f["rows"] for f in source["files"]) == 5349
    assert sum(f["bytes"] for f in source["files"]) == 1055835028
    assert "PENDING" in source["status"]
    assert source["native_framework"]["task"] == "inspect_evals/docvqa"
    assert adapter_spec("DOCVQA")["outcome_fields"] == ["answers"]
