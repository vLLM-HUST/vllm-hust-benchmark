"""Static command contract only; official candidates are never executed."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "bcb_plan", SCRIPTS / "prepare_bigcodebench_scoring_plan.py"
)
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def prepared_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "verify_source", lambda root: {"fixture": "identity"})
    rows = [
        {"task_id": f"BigCodeBench/{i}", "test": "private fixture"} for i in range(1140)
    ]
    ids = ["BigCodeBench/5", "BigCodeBench/8", "BigCodeBench/11"]
    payloads = {
        "original-dataset-private.jsonl": rows,
        "raw-solutions-private.jsonl": [
            {"task_id": i, "solution": "raise RuntimeError('never execute')"}
            for i in ids
        ],
    }
    hashes = {}
    for name, records in payloads.items():
        path = tmp_path / name
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
        )
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "source_id": "BIGCODEBENCH",
        "status": "AUTHOR_SANITIZER_INPUT_PREPARED_NOT_EXECUTED",
        "code_revision": plan.CODE_REVISION,
        "original_parquet_sha256_required_by_cli": plan.DATA_SHA,
        "original_dataset_rows_preserved": 1140,
        "original_solution_text_modified": False,
        "selected_task_ids": ids,
        "selected_original_row_indices": [5, 8, 11],
        "prepared_file_sha256": hashes,
    }
    (tmp_path / "receipt-private.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return receipt


def test_plan_preserves_author_protocol_without_execution(tmp_path, monkeypatch):
    prepared_fixture(tmp_path, monkeypatch)
    result = plan.build_plan(tmp_path, tmp_path)
    sanitize, evaluate = result["commands"]
    assert sanitize[2] == "bigcodebench.sanitize"
    assert (
        evaluate[evaluate.index("--selective_evaluate") + 1]
        == "BigCodeBench/5,BigCodeBench/8,BigCodeBench/11"
    )
    assert evaluate[evaluate.index("--no_gt") + 1] == "False"
    assert evaluate[evaluate.index("--calibrated") + 1] == "True"
    assert evaluate[evaluate.index("--max_as_limit") + 1] == "30720"
    assert "--parallel" not in evaluate
    assert result["required_environment"]["XDG_CACHE_HOME"] == "/output/fresh-cache"
    assert result["source_checkbox_eligible"] is False
    assert result["functional_scoring_executed"] is False
    scope = result["validation_scope"]
    assert scope["original_dataset_rows_preserved"] == 1140
    assert scope["selected_unique_tasks"] == 3
    assert scope["all_original_tasks_selected"] is False
    assert scope["sampling_does_not_remove_native_steps"] is True
    assert len(scope["required_per_selected_task"]) == 3
    assert "private fixture" not in json.dumps(result)


@pytest.mark.parametrize("failure", ["hash", "selection", "modified", "source"])
def test_invalid_preparation_or_source_rejected(tmp_path, monkeypatch, failure):
    receipt = prepared_fixture(tmp_path, monkeypatch)
    if failure == "hash":
        receipt["prepared_file_sha256"]["raw-solutions-private.jsonl"] = "wrong"
    elif failure == "selection":
        receipt["selected_original_row_indices"] = [0, 1, 2]
    elif failure == "modified":
        receipt["original_solution_text_modified"] = True
    else:

        def fail(root):
            raise ValueError("source identity")

        monkeypatch.setattr(plan, "verify_source", fail)
    (tmp_path / "receipt-private.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        plan.build_plan(tmp_path, tmp_path)
