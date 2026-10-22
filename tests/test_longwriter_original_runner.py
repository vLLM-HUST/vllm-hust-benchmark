"""CPU fixture tests, not long-output task execution or length quality evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_longwriter_original_samples.py"
spec = importlib.util.spec_from_file_location("longwriter_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_original_input_retained_and_final_reference_excluded():
    messages = [
        {"role": "user", "content": "原始任务，不改写\n"},
        {"role": "assistant", "content": "REFERENCE_ONLY"},
    ]
    assert runner.original_messages({"messages": messages}) == messages[:-1]
    assert messages[-1]["content"] == "REFERENCE_ONLY"


def test_missing_reference_or_generation_turn_is_not_invented():
    with pytest.raises(ValueError):
        runner.original_messages({"messages": [{"role": "user", "content": "x"}]})
    with pytest.raises(ValueError):
        runner.original_messages(
            {
                "messages": [
                    {"role": "assistant", "content": "history"},
                    {"role": "assistant", "content": "reference"},
                ]
            }
        )


def test_external_endpoint_refused_before_loading_data(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-root",
            str(tmp_path / "missing"),
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


def test_help_does_not_load_optional_data_package():
    assert (
        subprocess.run(
            [sys.executable, str(SCRIPT), "--help"], capture_output=True
        ).returncode
        == 0
    )


def test_executed_receipt_is_not_length_quality_or_different_benchmark():
    receipt = json.loads(
        (
            ROOT / "docs/workload-data/validation/longwriter-original-npu-20261022.json"
        ).read_text("utf-8")
    )
    assert (
        receipt["helper_sha256"]
        == hashlib.sha256(SCRIPT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    )
    assert receipt["unique_original_rows"] == 3 < receipt["original_dataset_rows"]
    assert receipt["is_longbench_write_evaluation"] is False
    assert receipt["length_requirement_qualification_claim"] is False
    assert receipt["quality_claim"] is False and receipt["full_suite_claim"] is False
    assert [row["completion_tokens"] for row in receipt["samples"]] == [
        2168,
        3064,
        5497,
    ]
    assert all(
        row["full_prompt_tokens"] == row["preflight_prompt_tokens"]
        and row["finish_reasons"] == ["stop"]
        for row in receipt["samples"]
    )
