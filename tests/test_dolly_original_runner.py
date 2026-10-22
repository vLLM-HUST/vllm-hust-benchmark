"""CPU fixtures only, not evidence of original dataset/model execution."""

import importlib.util
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_dolly_original_samples.py"
spec = importlib.util.spec_from_file_location("dolly_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_prefix_preserves_both_fields_and_excludes_reference_answer():
    constants = SimpleNamespace(
        PROMPT_WITH_INPUT_FORMAT="instruction:{instruction}\ncontext:{input}\nresponse:{response}END",
        PROMPT_NO_INPUT_FORMAT="instruction:{instruction}\nresponse:{response}END",
    )
    row = {
        "instruction": "  original {response}\n",
        "context": "原上下文\n",
        "response": "SCORER_ONLY",
    }
    result = runner.generation_prefix(row, constants)
    assert (
        result == "instruction:  original {response}\n\ncontext:原上下文\n\nresponse:"
    )
    assert "SCORER_ONLY" not in result
    assert (
        runner.generation_prefix({"instruction": "x", "context": ""}, constants)
        == "instruction:x\nresponse:"
    )


def test_external_endpoint_refused_before_private_data_access(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-root",
            str(tmp_path / "missing"),
            "--publisher-constants",
            str(tmp_path / "missing.py"),
            "--output",
            str(tmp_path / "output"),
            "--endpoint",
            "https://example.org/v1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "only local /v1 model endpoints" in result.stderr
    assert not (tmp_path / "output").exists()


def test_help_does_not_require_original_data_or_optional_package():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True
    )
    assert result.returncode == 0


def test_real_receipt_is_generation_integrity_not_quality_or_full_dataset():
    receipt = json.loads(
        (
            ROOT / "docs/workload-data/validation/dolly-original-npu-20261022.json"
        ).read_text("utf-8")
    )
    assert (
        receipt["helper_sha256"]
        == hashlib.sha256(SCRIPT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    )
    assert receipt["unique_original_rows"] == 3 < receipt["original_dataset_rows"]
    assert receipt["quality_claim"] is False and receipt["full_suite_claim"] is False
    assert receipt["publisher_reference_responses_present"] is True
    assert [s["row_index"] for s in receipt["samples"]] == [0, 1, 2]
    assert all(
        s["full_prompt_tokens"] == s["preflight_prompt_tokens"]
        and s["finish_reasons"] == ["stop"]
        for s in receipt["samples"]
    )
