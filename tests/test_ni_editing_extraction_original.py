"""Selection/reuse contracts only; not original task execution or score evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_ni_editing_extraction_original.py"
SPEC = importlib.util.spec_from_file_location("ni_editing_extraction_test", PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_help_has_only_two_added_tasks_without_optional_runtime_dependencies():
    result = subprocess.run(
        [sys.executable, str(PATH), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert all(name in result.stdout for name in module.TASKS)
    assert "task1575" not in result.stdout


def test_original_runner_default_tasks_and_small_task_limit_remain_unchanged():
    core = module.load_core()
    assert len(core.TASKS) == 3
    assert set(core.TASKS).isdisjoint(module.TASKS)
    original = core.TASKS
    try:
        core.TASKS = module.TASKS
        for name in module.TASKS:
            assert list(core.selected_indices(name, False)) == [0, 1, 2]
            with pytest.raises(ValueError, match="at most 250"):
                core.selected_indices(name, True)
    finally:
        core.TASKS = original


def test_selection_configuration_restored_even_on_failure(monkeypatch):
    original = {"original-fixture": ("0" * 64, 3)}

    def fail():
        assert core.TASKS is module.TASKS
        raise RuntimeError("unit fixture, no model call")

    core = SimpleNamespace(TASKS=original, main=fail)
    monkeypatch.setattr(module, "load_core", lambda: core)
    with pytest.raises(RuntimeError, match="unit fixture"):
        module.main()
    assert core.TASKS is original


def test_frozen_receipt_counts_six_unique_instances_not_handoff_repeats():
    receipt_path = (
        ROOT
        / "docs/workload-data/validation/ni-editing-extraction-original-npu-20261023.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert (
        receipt["public_runner_sha256"]
        == hashlib.sha256(PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    )
    assert receipt["reused_runner_sha256"] == module.CORE_SHA
    assert receipt["unique_ni_instances"] == 6
    assert receipt["total_model_calls_including_same_six_handoff_repeat"] == 12
    assert {
        r["task_file"]: (r["sha256"], r["source_instances"])
        for r in receipt["task_results"]
    } == module.TASKS
    for row in receipt["task_results"]:
        assert row["selected_original_indices"] == [0, 1, 2]
        assert row["all_normal_stop"] is True
        assert row["executed_unique"] == 3 < row["source_instances"]
    assert receipt["research_consumer_validated"] is False
    assert receipt["full_ni_or_underlying_wsc_btc_suite"] is False
    ledger = json.loads(
        (ROOT / "docs/workload-data/dataset-validation-status.json").read_text(
            encoding="utf-8"
        )
    )
    source = next(
        r for r in ledger["sources"] if r["source_id"] == "NATURAL_INSTRUCTIONS"
    )
    assert (
        source["primary_receipt_unique_instances"] + receipt["unique_ni_instances"]
        == source["unique_validated_instances_including_full_small_tasks"]
    )
    assert source["checked"] is False
