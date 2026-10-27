"""Invented parser fixtures only; not HumanEval tasks or functional scores."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from vllm_hust_workload_data.adapters import adapt_row
from vllm_hust_workload_data.registry import adapter_spec


def test_original_prompt_whitespace_and_scorer_only_fields():
    row = {
        "task_id": "fixture/0",
        "prompt": "def fixture():\n    ",
        "canonical_solution": "SCORER_ONLY_SOLUTION",
        "test": "SCORER_ONLY_TEST",
        "entry_point": "SCORER_ONLY_ENTRY",
    }
    record = adapt_row(
        "HUMANEVAL",
        row,
        source={"source_id": "HUMANEVAL"},
        spec=adapter_spec("HUMANEVAL"),
        index=0,
    )
    assert record.task["task_id"] == row["task_id"]
    assert record.task["input"] == {"prompt": row["prompt"]}
    assert "SCORER_ONLY" not in json.dumps(record.input_record())
    assert record.outcome_record() == {
        k: row[k] for k in ("canonical_solution", "test", "entry_point")
    }
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_humaneval_original_generation.py"
    )
    spec = importlib.util.spec_from_file_location("humaneval_generation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = module.completion_payload(record, "fixture-model")
    assert payload["prompt"] == row["prompt"]
    assert "SCORER_ONLY" not in json.dumps(payload)
    assert set(payload) == {"prompt", "model", "temperature", "max_tokens", "n"}


@pytest.mark.parametrize(
    "field", ["prompt", "canonical_solution", "test", "entry_point"]
)
def test_missing_original_fields_rejected(field):
    row = {
        "task_id": "fixture/0",
        "prompt": "fixture",
        "canonical_solution": "x",
        "test": "y",
        "entry_point": "z",
    }
    del row[field]
    with pytest.raises(ValueError, match="missing required fields"):
        adapt_row(
            "HUMANEVAL",
            row,
            source={"source_id": "HUMANEVAL"},
            spec=adapter_spec("HUMANEVAL"),
            index=0,
        )


def test_generation_help_and_external_endpoint_refusal(tmp_path):
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_humaneval_original_generation.py"
    )
    assert (
        subprocess.run(
            [sys.executable, str(script), "--help"], capture_output=True
        ).returncode
        == 0
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--data-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "not-created"),
            "--model",
            "fixture",
            "--endpoint",
            "https://example.com/v1",
        ],
        capture_output=True,
    )
    assert result.returncode == 2
    assert not (tmp_path / "not-created").exists()
