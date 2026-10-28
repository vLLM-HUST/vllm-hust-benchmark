"""Structural fixtures only; not publisher task or native run evidence."""

import json

import pytest

from vllm_hust_workload_data.appworld import SPLITS, adapt_case, original_task_bundles


def fixture_root(tmp_path):
    (tmp_path / "datasets").mkdir()
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
