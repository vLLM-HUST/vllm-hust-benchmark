"""Lossless task/setup/reference fixtures, never native GUI execution."""

import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "workload-preparation/src")
)
from vllm_hust_workload_data.osworld import original_task_packages
from vllm_hust_workload_data.osworld import pipeline_manifest
from vllm_hust_workload_data.pipeline import plan, run, prepared_rows
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.registry import adapter_spec


@pytest.fixture
def data(tmp_path):
    files = {
        "evaluation_examples/test_all.json": {"chrome": ["case-b", "case-a"]},
        "evaluation_examples/examples/chrome/case-b.json": dict(
            id="case-b",
            instruction="original",
            config=[dict(type="original-setup", parameters=dict(urls=["original"]))],
            evaluator=dict(
                func="original_metric", expected=dict(original_gold=["original"])
            ),
        ),
        "evaluation_examples/examples/chrome/case-a.json": dict(
            id="case-a",
            instruction="second original",
            evaluator=dict(func="second-original"),
        ),
    }
    records = []
    for name, value in files.items():
        raw = json.dumps(value).encode()
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        records.append(
            dict(path=name, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        )
    return (
        tmp_path,
        dict(
            source_id="OSWORLD",
            repository="original-fixture",
            revision="fixture",
            files=records,
        ),
        files,
    )


def test_order_and_native_relationships_are_lossless(data):
    root, pins, originals = data
    rows = list(original_task_packages(root, manifest=pins))
    assert [r["task_id"] for r in rows] == ["case-b", "case-a"]
    assert (
        "config" not in rows[1]["input"]
    )  # Original no-setup task, not a fabricated empty config.
    for row in rows:
        assert "evaluator" not in row["input"]
        reconstructed = {**row["input"], "evaluator": row["outcome"]["evaluator"]}
        assert reconstructed == originals[row["original_asset"]["path"]]
        assert row["native_execution_complete"] is False


def test_modified_task_rejected_before_read(data):
    root, pins, _ = data
    (root / "evaluation_examples/examples/chrome/case-b.json").write_text("{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        list(original_task_packages(root, manifest=pins))


def test_modified_catalog_rejected_when_iterator_closes(data):
    root, pins, _ = data
    rows = original_task_packages(root, manifest=pins)
    next(rows)
    (root / "evaluation_examples/test_all.json").write_text("{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        rows.close()


def test_unified_pipeline_and_unchanged_consumer_preserve_native_package(data):
    root, pins, originals = data
    pins["revision"] = adapter_spec("OSWORLD")["code_revision"]
    binding = root.parent / (root.name + "-binding.json")
    binding.write_text(json.dumps(pipeline_manifest(pins)))
    config = {
        "sources": [
            dict(source_id="OSWORLD", data_root=str(root), manifest=str(binding))
        ]
    }
    assert plan(config)["sources"][0]["ready_to_attempt"] is True
    output = root.parent / (root.name + "-index")
    report = run(config, output=output)
    assert report["model_calls"] == 0 and report["scoring"] == "NOT_RUN"
    database = output / "workload.sqlite"
    rows = list(prepared_rows(database))
    assert len(rows) == 2
    for index, row in enumerate(rows):
        record = resolve_input(database, row["record_id"], data_root=root)
        name = ["case-b", "case-a"][index]
        expected = originals[f"evaluation_examples/examples/chrome/{name}.json"]
        assert {
            **record.task["input"],
            "evaluator": record.outcome_record()["evaluator"],
        } == expected
        assert "evaluator" not in record.task["input"]
        with pytest.raises(ValueError, match="not a ready model request"):
            record.request_body()
    (root / "evaluation_examples/examples/chrome/case-b.json").write_text("{}")
    with pytest.raises(ValueError, match="identity mismatch"):
        resolve_input(database, rows[0]["record_id"], data_root=root)
