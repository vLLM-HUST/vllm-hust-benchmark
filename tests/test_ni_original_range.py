"""CPU selection contracts; not native task execution evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_ni_original_range.py"
sys.path.insert(0, str(PATH.parent))
try:
    spec = importlib.util.spec_from_file_location("ni_range_test", PATH)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
finally:
    sys.path.pop(0)


def test_default_tail_full_preserve_original_indices():
    assert runner.selected_indices(6493) == [0, 1, 2]
    assert runner.selected_indices(6493, 6490, 3) == [6490, 6491, 6492]
    assert runner.selected_indices(261, full=True) == list(range(261))
    assert len(runner.selected_indices(6493, full=True)) == 6493


@pytest.mark.parametrize(
    "start,count,full", [(-1, 3, False), (260, 2, False), (0, 0, False), (0, 3, True)]
)
def test_invalid_or_ambiguous_selection_rejected(start, count, full):
    with pytest.raises(ValueError):
        runner.selected_indices(261, start, count, full)


def test_original_pinned_core_and_five_registered_tasks():
    core = runner.load_core()
    assert len({**core.TASKS, **runner.EXTRA_TASKS}) == 5


def test_actual_range_receipt_preserves_pinned_core_and_original_protocol():
    receipt = json.loads(
        (
            PATH.parents[1]
            / "docs/workload-data/validation/ni-native-range-audit-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["new_original_native_tasks"] == 9
    assert receipt["audit_new_generation_calls"] == 0
    assert receipt["audit_model_tokenizer_requests"] == 9
    assert not receipt["source_checkbox_eligible"]
    for row in receipt["checks"]:
        assert row["actual_tasks"] == len(row["selected_original_indices"]) == 3
        assert row["unmodified_original_collator_prompts_equal"]
        assert row["original_scorer_recomputed_equal"]
        assert (
            row["driver_sha256"]
            == hashlib.sha256(PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        )


def test_complete_extraction_coverage_is_disjoint_and_not_entire_ni():
    receipt = json.loads(
        (
            PATH.parents[1]
            / "docs/workload-data/validation/ni-extraction-complete-audit-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["original_task_size"] == receipt["unique_original_tasks"] == 261
    covered = []
    for batch in receipt["batches"]:
        start, end = batch["original_interval_inclusive"]
        assert end - start + 1 == batch["original_tasks"]
        covered.extend(range(start, end + 1))
        assert batch["original_scorer_recomputed_equal"]
        assert batch["original_collator_requests_and_response_integrity_verified"]
    assert len(covered) == len(set(covered)) == 261
    assert set(covered) == set(range(261))
    assert not receipt["full_ni_suite_validated"]
    assert receipt["audit_new_generation_calls"] == 0


def test_registered_ni_delivery_counts_distinct_tasks_and_limits():
    docs = PATH.parents[1] / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    item = ledger["delivery_inventory"]["NATURAL_INSTRUCTIONS"]
    receipt = json.loads(
        (PATH.parents[1] / item["delivery_receipt"]).read_text("utf-8")
    )
    assert item["checked"] and item["status"] == "AVAILABLE"
    complete = sum(task["tasks"] for task in receipt["complete_original_task_files"])
    sampled = sum(
        len(task["original_indices"]) for task in receipt["sampled_original_task_files"]
    )
    assert complete + sampled == receipt["verified_unique_original_instances"] == 1797
    assert receipt["registered_original_instances"] == 8284
    assert not receipt["full_ni_suite_validated"]
    assert not receipt["all_registered_instances_executed"]
    assert not receipt["underlying_dataset_benchmark_equivalence"]
    for name in receipt["native_execution_receipts"]:
        assert (docs / "validation" / name).is_file()


def test_help_without_optional_original_runtime():
    result = subprocess.run(
        [sys.executable, str(PATH), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--full" in result.stdout and "--start" in result.stdout
