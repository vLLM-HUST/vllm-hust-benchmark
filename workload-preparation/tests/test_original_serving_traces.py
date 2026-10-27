"""Invented metadata fixtures, not production trace data or performance evidence."""

import csv
import json

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.registry import source_spec


@pytest.mark.parametrize("source", ["AZURE_LLM", "MOONCAKE"])
def test_original_metadata_roundtrip_never_becomes_model_request(tmp_path, source):
    root = tmp_path / "data"
    root.mkdir()
    if source == "AZURE_LLM":
        originals = [
            {
                "TIMESTAMP": f"2023-11-11 00:00:0{i}",
                "ContextTokens": str(10 + i),
                "GeneratedTokens": "2",
            }
            for i in range(3)
        ]
        path = root / "fixture.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(originals[0]))
            writer.writeheader()
            writer.writerows(originals)
    else:
        originals = [
            {
                "timestamp": i * 100,
                "input_length": 1024 + i,
                "output_length": 2,
                "hash_ids": [7, 11 + i],
            }
            for i in range(3)
        ]
        path = root / "fixture.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in originals), encoding="utf-8"
        )
    binding = bind_source(
        source, data_root=root, files=[path.name], revision="invented-unit-fixture"
    )
    records = list(iter_workload(source, data_root=root, manifest=binding))
    assert len(records) == 3
    for record, original in zip(records, originals):
        assert record.task["input"] == {}
        assert record.replay["annotations"] == original
        assert record.outcome_record() is None
        assert record.runtime["kind"] == "request_metadata"
        with pytest.raises(ValueError):
            record.request_body()
    manifest = tmp_path / "binding.json"
    manifest.write_text(json.dumps(binding), encoding="utf-8")
    run(
        {
            "sources": [
                {"source_id": source, "data_root": str(root), "manifest": str(manifest)}
            ]
        },
        output=tmp_path / "index",
    )
    db = tmp_path / "index/workload.sqlite"
    references = list(prepared_rows(db))
    assert len(references) == 3
    for record, reference in zip(records, references):
        assert (
            resolve_input(db, reference["record_id"], data_root=root).input_record()
            == record.input_record()
        )


@pytest.mark.parametrize("source", ["AZURE_LLM", "MOONCAKE"])
def test_trace_catalog_retains_fixed_source_urls_and_original_units(source):
    manifest = source_spec(source)
    assert len(manifest["files"]) == 2
    assert manifest["unit"] == "original_request_metadata_records"
    for file in manifest["files"]:
        assert (
            file["url"]
            == "https://raw.githubusercontent.com/"
            + manifest["repo_id"]
            + "/"
            + manifest["revision"]
            + "/"
            + file["upstream_path"]
        )
        assert file["rows"] > 3
