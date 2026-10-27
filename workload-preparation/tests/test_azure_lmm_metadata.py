"""Invented tiny metadata fixtures, never original request/performance evidence."""

import csv
from dataclasses import asdict
import gzip
import hashlib
import json

import pytest

from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def test_gzip_trace_full_roundtrip_has_no_prompt_or_media(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    path = data / "fixture.csv.gz"
    rows = [
        {
            "TIMESTAMP": f"2024-10-15 00:00:0{i}",
            "NumImages": str(i),
            "ContextTokens": "0002",
            "GeneratedTokens": "1",
        }
        for i in range(3)
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    catalog = tmp_path / "catalog"
    resources = catalog / "data/catalogs"
    resources.mkdir(parents=True)
    source = {
        **source_spec("AZURE_LMM_2025"),
        "revision": "invented-unit-fixture",
        "files": [
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "role": "raw_source",
            }
        ],
    }
    (resources / "source_manifests_v1.json").write_text(
        json.dumps({"sources": {"AZURE_LMM_2025": source}})
    )
    (resources / "adapter_registry_v1.json").write_text(
        json.dumps({"adapters": {"AZURE_LMM_2025": adapter_spec("AZURE_LMM_2025")}})
    )
    binding = bind_source(
        "AZURE_LMM_2025",
        data_root=data,
        files=[path.name],
        revision=source["revision"],
        catalog_root=catalog,
    )
    records = list(
        iter_workload(
            "AZURE_LMM_2025", data_root=data, manifest=binding, catalog_root=catalog
        )
    )
    assert len(records) == 3
    for record, raw in zip(records, rows):
        assert record.replay["annotations"] == raw and record.task["input"] == {}
        assert (
            record.runtime["kind"] == "request_metadata"
            and record.outcome_record() is None
        )
        with pytest.raises(ValueError, match="not a ready model request"):
            record.request_body()
    manifest = tmp_path / "binding.json"
    manifest.write_text(json.dumps(binding))
    run(
        {
            "sources": [
                {
                    "source_id": "AZURE_LMM_2025",
                    "data_root": str(data),
                    "manifest": str(manifest),
                }
            ]
        },
        output=tmp_path / "index",
        catalog_root=catalog,
    )
    db = tmp_path / "index/workload.sqlite"
    for record, reference in zip(records, prepared_rows(db)):
        assert asdict(
            resolve_input(db, reference["record_id"], data_root=data)
        ) == asdict(record)


@pytest.mark.parametrize("value", [-1, True, "-1", "2.0", "²"])
def test_invalid_image_counts_rejected_not_coerced(value):
    row = {
        "TIMESTAMP": "source timestamp",
        "NumImages": value,
        "ContextTokens": "1",
        "GeneratedTokens": "2",
    }
    with pytest.raises(ValueError, match="nonnegative integer"):
        adapt_row(
            "AZURE_LMM_2025",
            row,
            source={"source_id": "AZURE_LMM_2025", "revision": "fixture"},
            spec=adapter_spec("AZURE_LMM_2025"),
            index=0,
        )


def test_unknown_extra_field_cannot_silently_disappear():
    row = {
        "TIMESTAMP": "source timestamp",
        "NumImages": "1",
        "ContextTokens": "1",
        "GeneratedTokens": "2",
        "unknown": "not discarded",
    }
    with pytest.raises(ValueError, match="four-field schema"):
        adapt_row(
            "AZURE_LMM_2025",
            row,
            source={"source_id": "AZURE_LMM_2025", "revision": "fixture"},
            spec=adapter_spec("AZURE_LMM_2025"),
            index=0,
        )
