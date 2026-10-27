"""Schema fixtures, not original tasks or native execution evidence."""

import json

import pytest

from vllm_hust_workload_data.adapters import adapt_row, iter_source_rows
from vllm_hust_workload_data.registry import adapter_spec, source_spec


@pytest.mark.parametrize(
    "name,original,identity",
    [
        (
            "level_1_query_meta.json",
            {"id": 0, "query": "fixture shopping goal"},
            "shopping:1:0",
        ),
        (
            "travelplanning_query_en.json",
            {
                "id": 0,
                "query": "fixture travel goal",
                "meta_info": {"reference": "keep private"},
                "query_with_constraints": "reference-only fixture",
            },
            "travel:en:0",
        ),
    ],
)
def test_all_original_fields_preserved_reference_not_input(
    tmp_path, name, original, identity
):
    file = tmp_path / name
    file.write_text(json.dumps([original]), encoding="utf-8")
    spec = adapter_spec("DEEPPLANNING")
    row = next(iter_source_rows(file, spec))
    record = adapt_row(
        "DEEPPLANNING", row, source={"source_id": "DEEPPLANNING"}, spec=spec, index=0
    )
    assert record.task["task_id"] == identity
    reconstructed = (
        record.task["input"] | record.outcome["original_evaluation_metadata"]
    )
    assert reconstructed == original
    assert "reference-only fixture" not in json.dumps(record.input_record())
    assert "keep private" not in json.dumps(record.input_record())
    assert record.replay["annotations"]["database_archive"].startswith("databases/")
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_unknown_fields_and_duplicate_ids_rejected(tmp_path):
    file = tmp_path / "level_1_query_meta.json"
    file.write_text(json.dumps([{"id": 0, "query": "fixture", "unexpected": 1}]))
    with pytest.raises(ValueError, match="unknown original fields"):
        list(iter_source_rows(file, adapter_spec("DEEPPLANNING")))
    file.write_text(json.dumps([{"id": 0, "query": "fixture"}] * 2))
    with pytest.raises(ValueError, match="duplicate"):
        list(iter_source_rows(file, adapter_spec("DEEPPLANNING")))


def test_registry_original_360_rows_and_five_database_archives():
    manifest = source_spec("DEEPPLANNING")
    assert (
        sum(file["rows"] for file in manifest["files"] if file["role"] == "data") == 360
    )
    assert sum(file["role"] == "native_environment" for file in manifest["files"]) == 5
