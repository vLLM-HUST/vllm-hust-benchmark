"""Structural fixtures only; not publisher task or native run evidence."""

import json

import pytest

from vllm_hust_workload_data.appworld import SPLITS, adapt_case, original_task_bundles
from vllm_hust_workload_data import iter_workload
from vllm_hust_workload_data.bindings import bind_source
from vllm_hust_workload_data.consumer import resolve_input
from vllm_hust_workload_data.pipeline import prepared_rows, run


def fixture_root(tmp_path):
    (tmp_path / "datasets").mkdir()
    for name in ("version.txt", "LICENSE", "README_BEFORE_SHARING.md"):
        (tmp_path / name).write_text("structural fixture only")
    for name in ("base_dbs", "api_docs"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "fixture.json").write_text("{}")
    for index, split in enumerate(SPLITS):
        identity = f"fixture_{index}"
        (tmp_path / "datasets" / f"{split}.txt").write_text(identity + "\n")
        folder = tmp_path / "tasks" / identity
        (folder / "dbs").mkdir(parents=True)
        (folder / "ground_truth").mkdir()
        (folder / "specs.json").write_text(
            json.dumps(
                {"instruction": "fixture only", "extra_original_field": {"keep": True}}
            )
        )
        (folder / "dbs" / "fixture.jsonl").write_text("{}\n")
        (folder / "ground_truth" / "evaluation.py").write_text(
            'raise RuntimeError("must never execute during reading")\n'
        )
    return tmp_path


def test_original_fields_order_and_reference_separation(tmp_path):
    root = fixture_root(tmp_path)
    rows = list(original_task_bundles(root))
    assert [row["split"] for row in rows] == list(SPLITS)
    assert rows[0]["original_specs"]["extra_original_field"] == {"keep": True}
    record = adapt_case(rows[0], source={"source_id": "APPWORLD"}, index=0)
    assert "reference_files" not in record.task["input"]
    assert record.outcome["reference_files"][0]["path"].endswith("evaluation.py")
    assert record.runtime["execution"] == "not_executed"
    assert not record.replay["synthetic_fallback"]
    assert set(record.task["input"]["shared_environment_files"]) == {
        "base_dbs",
        "api_docs",
    }


def test_overlapping_tasks_rejected_before_yield(tmp_path):
    root = fixture_root(tmp_path)
    (root / "datasets/dev.txt").write_text("fixture_0\n")
    with pytest.raises(ValueError, match="overlapping"):
        next(original_task_bundles(root))


def test_missing_reference_bundle_is_not_synthetic_fallback(tmp_path):
    root = fixture_root(tmp_path)
    (root / "tasks/fixture_0/ground_truth/evaluation.py").unlink()
    with pytest.raises(ValueError, match="reference bundles"):
        next(original_task_bundles(root))


def test_unified_binding_reader_and_reference_integrity(tmp_path):
    root = fixture_root(tmp_path)
    manifest = bind_source(
        "APPWORLD",
        data_root=root,
        files=["datasets/test_normal.txt"],
        revision="fixture-local-not-publisher-verified",
    )
    records = list(iter_workload("APPWORLD", data_root=root, manifest=manifest))
    assert len(records) == 1 and records[0].task["task_id"] == "fixture_2"
    assert not manifest["upstream_revision_verified"]
    assert len(manifest["native_task_assets"]) == 9
    (root / "tasks/fixture_2/ground_truth/evaluation.py").write_text(
        "# changed reference\n"
    )
    with pytest.raises(ValueError):
        list(iter_workload("APPWORLD", data_root=root, manifest=manifest))


def test_index_consumer_preserves_bundle_and_rejects_changed_database(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    fixture_root(root)
    manifest = bind_source(
        "APPWORLD",
        data_root=root,
        files=["datasets/test_normal.txt"],
        revision="fixture-local-not-publisher-verified",
    )
    manifest_path = tmp_path / "binding.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "prepared"
    report = run(
        {
            "sources": [
                {
                    "source_id": "APPWORLD",
                    "data_root": str(root),
                    "manifest": str(manifest_path),
                }
            ]
        },
        output=output,
    )
    assert report["native_tasks_run"] == 0
    indexed = list(prepared_rows(output / "workload.sqlite"))
    assert len(indexed) == 1
    record = resolve_input(
        output / "workload.sqlite", indexed[0]["record_id"], data_root=root
    )
    assert record.task["task_id"] == "fixture_2"
    assert record.outcome["reference_files"][0]["path"].endswith("evaluation.py")
    (root / "tasks/fixture_2/dbs/fixture.jsonl").write_text('{"modified": true}\n')
    with pytest.raises(ValueError):
        resolve_input(
            output / "workload.sqlite", indexed[0]["record_id"], data_root=root
        )
