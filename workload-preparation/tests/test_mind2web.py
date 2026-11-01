from copy import deepcopy
import json

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.mind2web import restore_original_task
from vllm_hust_workload_data.registry import adapter_spec
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.pipeline import run, prepared_rows
from vllm_hust_workload_data.consumer import resolve_input


def test_original_task_roundtrip_and_reference_isolation():
    row = {
        "annotation_id": "fixture-id",
        "website": "site",
        "domain": "domain",
        "subdomain": "sub",
        "confirmed_task": "original fixture task",
        "extension": {"preserve": [1, 2]},
        "action_reprs": ["reference action"],
        "actions": [
            {
                "action_uid": "action-1",
                "raw_html": "<html/>",
                "cleaned_html": "<html/>",
                "operation": {"op": "CLICK", "value": ""},
                "pos_candidates": [{"backend_node_id": "1"}],
                "neg_candidates": [],
                "extension": True,
            }
        ],
    }
    before = deepcopy(row)
    record = adapt_row(
        "MIND2WEB",
        row,
        source={"source_id": "MIND2WEB"},
        spec=adapter_spec("MIND2WEB"),
        index=0,
    )
    assert restore_original_task(record) == before
    assert row == before
    assert "operation" not in record.task["input"]["pages"][0]
    assert "pos_candidates" not in record.task["input"]["pages"][0]
    assert "action_reprs" not in record.task["input"]
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_original_json_binding_index_and_consumer_roundtrip(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    row = {
        "annotation_id": "fixture-index",
        "website": "site",
        "domain": "domain",
        "subdomain": "sub",
        "confirmed_task": "original task fixture",
        "action_reprs": ["reference only"],
        "actions": [
            {
                "action_uid": "action",
                "raw_html": "<html/>",
                "cleaned_html": "<html/>",
                "operation": {"op": "CLICK", "value": ""},
                "pos_candidates": [],
                "neg_candidates": [],
            }
        ],
    }
    (root / "original.json").write_text(json.dumps([row]), encoding="utf-8")
    manifest = bind_source(
        "MIND2WEB",
        data_root=root,
        files=["original.json"],
        revision="fixture-original-snapshot",
    )
    binding = tmp_path / "manifest.json"
    binding.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "prepared"
    run(
        {
            "sources": [
                {
                    "source_id": "MIND2WEB",
                    "data_root": str(root),
                    "manifest": str(binding),
                }
            ]
        },
        output=output,
    )
    database = output / "workload.sqlite"
    rows = list(prepared_rows(database))
    assert len(rows) == 1
    restored = resolve_input(database, rows[0]["record_id"], data_root=root)
    assert restore_original_task(restored) == row
