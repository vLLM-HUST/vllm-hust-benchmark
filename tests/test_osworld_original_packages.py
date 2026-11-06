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
