"""Artificial source-shaped fields only, not downloaded publisher tasks."""

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def test_all_original_fields_stay_in_input_outcome_or_annotations():
    row = {
        "id": "fixture-id",
        "question": "fixture question",
        "ideal": "fixture reference",
        "distractors": ["fixture alternative"],
        "canary": "fixture canary",
        "figure-path": "images/original.png",
        "table-path": ["images/table.png"],
        "protocol": "fixture original context",
        "sources": ["fixture source"],
        "source": "fixture source",
        "paper-title": "fixture title",
        "key-passage": None,
        "is_opensource": True,
        "tag": "fixture tag",
        "version": "fixture version",
    }
    spec = adapter_spec("LABBENCH")
    record = adapt_row(
        "LABBENCH",
        row,
        source={"source_id": "LABBENCH", "file": "source/FigQA/fixture-public.jsonl"},
        spec=spec,
        index=0,
    )
    assert record.outcome_record() == {"ideal": row["ideal"]}
    assert "ideal" not in record.task["input"]
    reconstructed = {
        **record.task["input"],
        **record.replay["annotations"],
        **record.outcome_record(),
    }
    assert reconstructed == row
    assert record.task["input"]["figure-path"] == "images/original.png"
    assert record.task["input"]["protocol"] == row["protocol"]


def test_fixed_original_public_release_inventory():
    source = source_spec("LABBENCH")
    assert len(source["files"]) == 31
    assert sum(item["rows"] for item in source["files"]) == 1967
    assert all(
        item["role"] == "data" and item["path"].endswith("-public.jsonl")
        for item in source["files"]
    )
