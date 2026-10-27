"""CPU artifact/entry checks only, not BFCL task execution or scores."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_native_handoff_retains_static_scoring_boundary_and_real_identity():
    record = json.loads(
        (
            ROOT
            / "docs/workload-data/validation/bfcl-v4-simple-python-public-handoff-npu-20261023.json"
        ).read_text(encoding="utf-8")
    )
    assert record["status"] == "END_TO_END_EXECUTED_AND_SCORED"
    assert record["selected_samples"] == [
        "simple_python_0",
        "simple_python_1",
        "simple_python_2",
    ]
    assert record["unique_original_tasks"] == 3
    assert record["original_task_rows"] == 400
    assert record["official_valid_count"] == 3
    assert record["unchanged_official_python_files"] == 110
    assert record["cpu_model_inference"] is False
    assert record["generated_code_or_tool_execution"] is False
    assert record["full_suite_claim"] is record["performance_claim"] is False
    assert (
        record["scoring_kind"] == "ORIGINAL_BFCL_AST_ARGUMENT_MATCH_NOT_TOOL_EXECUTION"
    )
    assert record["model_deployment_adaptation"]["underscore_to_dot"] is False
    assert all(
        s["finish_reason"] == "stop"
        and s["usage"]["prompt_tokens"] == s["preflight_prompt_tokens"]
        for s in record["samples"]
    )
    assert record["samples"][1]["official_valid"] is True
    assert (
        record["samples"][1]["official_error_type"] == "simple_function_checker:unclear"
    )
    assert (
        record["helper_sha256"]
        == "f90797827d70533ad778bf97a4a7c8c498e8103c68d533edd2fc28b2e4734c44"  # pragma: allowlist secret - public historical runner SHA
    )


def test_original_ranges_and_full_do_not_substitute_or_duplicate_tasks():
    path = ROOT / "scripts/run_bfcl_v4_simple_python_original.py"
    spec = importlib.util.spec_from_file_location("bfcl_native_selection", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    assert runner.selected_indices() == [0, 1, 2]
    assert runner.selected_indices(397, 3) == [397, 398, 399]
    assert runner.selected_indices(full=True) == list(range(400))
    for start, count in [(-1, 3), (399, 2), (0, 0)]:
        with pytest.raises(ValueError):
            runner.selected_indices(start, count)


def test_public_help_without_installed_bfcl_or_model_dependencies():
    script = ROOT / "scripts/run_bfcl_v4_simple_python_original.py"
    assert (
        subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True
        ).returncode
        == 0
    )


def test_new_edge_audit_is_not_full_simple_python_or_bfcl_claim():
    receipt = json.loads(
        (
            ROOT / "docs/workload-data/validation/bfcl-native-edges-audit-20261024.json"
        ).read_text("utf-8")
    )
    assert receipt["unique_native_tasks_audited"] == 6
    assert (
        receipt["all_original_task_fields_checked"]
        == receipt["fixed_ci_indexed_tasks"]
        == 400
    )
    assert receipt["original_prepare_resolve_roundtrips"] == 6
    assert receipt["official_runtime_python_files_verified"] == 110
    assert receipt["audit_generation_calls"] == 0
    assert not receipt["full_simple_python_validated"]
    assert not receipt["full_bfcl_suite_validated"]
    assert not receipt["generated_code_or_tools_executed"]
    script = ROOT / "scripts/run_bfcl_v4_simple_python_original.py"
    assert (
        receipt["current_runner_sha256"]
        == hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    )
    assert [item["original_interval_inclusive"] for item in receipt["batches"]] == [
        [0, 2],
        [397, 399],
    ]


def test_complete_registered_category_is_not_full_bfcl_or_tool_execution():
    docs = ROOT / "docs/workload-data"
    ledger = json.loads((docs / "dataset-validation-status.json").read_text("utf-8"))
    item = ledger["delivery_inventory"]["BFCL"]
    delivery = json.loads((ROOT / item["delivery_receipt"]).read_text("utf-8"))
    receipt = json.loads(
        (docs / "validation" / delivery["native_execution_receipt"]).read_text("utf-8")
    )
    assert item["checked"] and item["status"] == "AVAILABLE"
    covered = []
    for batch in receipt["batches"]:
        start, end = batch["original_interval_inclusive"]
        assert end - start + 1 == batch["actual_original_tasks"]
        covered.extend(range(start, end + 1))
        assert batch["official_ast_score_recomputed_equal"]
        assert batch["full_original_handler_prompts_equal"]
    assert (
        len(covered)
        == len(set(covered))
        == receipt["unique_native_tasks_audited"]
        == 400
    )
    assert set(covered) == set(range(400))
    assert receipt["original_prepare_resolve_roundtrips"] == 400
    assert receipt["full_simple_python_validated"]
    assert not delivery["full_bfcl_suite_validated"]
    assert not delivery["live_tools_executed"]


def test_optional_dependency_snapshot_has_cpu_import_runtime_not_cuda_stack():
    raw = (
        ROOT / "configs/workload-native/bfcl-simple-python-py312-dependencies.txt"
    ).read_text(encoding="utf-8")
    requirements = [
        line for line in raw.splitlines() if line and not line.startswith("#")
    ]
    assert "torch==2.8.0+cpu" in requirements
    assert "soundfile==0.13.1" in requirements
    assert not any(
        line.lower().startswith(("nvidia", "cuda", "vllm", "sglang"))
        for line in requirements
    )
