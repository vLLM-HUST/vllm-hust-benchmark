"""CPU artifact/entry checks only, not BFCL task execution or scores."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

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
    script = ROOT / "scripts/run_bfcl_v4_simple_python_original.py"
    assert (
        record["helper_sha256"]
        == hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    )


def test_public_help_without_installed_bfcl_or_model_dependencies():
    script = ROOT / "scripts/run_bfcl_v4_simple_python_original.py"
    assert (
        subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True
        ).returncode
        == 0
    )


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
