"""Artificial source-shaped snippets only; no publisher code data or NPU run."""

from copy import deepcopy
import hashlib
import json

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.adapters import iter_workload
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def test_original_context_and_reference_boundaries():
    row = {
        "repo_name": "fixture/repo",
        "file_path": "fixture.py",
        "context": [
            {
                "path": "other.py",
                "snippet": "fixture snippet",
                "extra": "original extra",
            }
        ],
        "import_statement": "fixture import",
        "cropped_code": "fixture prefix",
        "next_line": "fixture answer",
        "all_code": "fixture prefix plus answer",
        "gold_snippet_index": 0,
        "created_at": "2023-12-02",
        "level": "2k",
        "token_num": 17,
    }
    record = adapt_row(
        "REPOBENCH",
        row,
        source={
            "source_id": "REPOBENCH",
            "file": "python/data/fixture.parquet",
            "file_sha256": "fixture-file-identity",
        },
        spec=adapter_spec("REPOBENCH"),
        index=0,
    )
    assert record.task["input"]["context"] == row["context"]
    assert not {"next_line", "all_code", "gold_snippet_index"} & set(
        record.task["input"]
    )
    assert {
        **record.task["input"],
        **record.outcome_record(),
        **record.replay["annotations"],
    } == row


def test_all_original_language_files_registered_separately():
    source = source_spec("REPOBENCH")
    assert len(source["files"]) == 10
    assert sum(item["rows"] for item in source["files"]) == 49684
    assert {item["language"] for item in source["files"]} == {"python", "java"}
    assert all(
        item["role"] == "raw_source" and item["dataset_revision"] in item["url"]
        for item in source["files"]
    )


def test_binding_preserves_language_dataset_revision_and_rejects_relabeling(tmp_path):
    row = {
        "repo_name": "fixture/repo",
        "file_path": "fixture.py",
        "context": [],
        "import_statement": "fixture import",
        "cropped_code": "fixture prefix",
        "next_line": "fixture target",
        "all_code": "fixture full code",
        "gold_snippet_index": None,
        "created_at": "2023-12-02",
        "level": "2k",
        "token_num": 7,
    }
    data = tmp_path / "data"
    path = data / "python/data/fixture.jsonl"
    path.parent.mkdir(parents=True)
    payload = (json.dumps(row) + "\n").encode()
    path.write_bytes(payload)
    source = deepcopy(source_spec("REPOBENCH"))
    source["files"] = [
        {
            "path": "python/data/fixture.jsonl",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "rows": 1,
            "role": "raw_source",
            "language": "python",
            "dataset_repo": "fixture/publisher",
            "dataset_revision": "fixture-data-revision",
        }
    ]
    catalog = tmp_path / "catalog/data/catalogs"
    catalog.mkdir(parents=True)
    (catalog / "source_manifests_v1.json").write_text(
        json.dumps({"sources": {"REPOBENCH": source}})
    )
    (catalog / "adapter_registry_v1.json").write_text(
        json.dumps({"adapters": {"REPOBENCH": adapter_spec("REPOBENCH")}})
    )
    binding = bind_source(
        "REPOBENCH",
        data_root=data,
        files=["python/data/fixture.jsonl"],
        revision=source["revision"],
        catalog_root=catalog.parent.parent,
    )
    record = list(
        iter_workload(
            "REPOBENCH",
            data_root=data,
            manifest=binding,
            catalog_root=catalog.parent.parent,
        )
    )[0]
    assert record.source["language"] == "python"
    assert record.source["dataset_repo"] == "fixture/publisher"
    assert record.source["dataset_revision"] == "fixture-data-revision"
    binding["files"][0]["language"] = "java"
    with pytest.raises(ValueError, match="provenance"):
        list(
            iter_workload(
                "REPOBENCH",
                data_root=data,
                manifest=binding,
                catalog_root=catalog.parent.parent,
            )
        )
