"""Offline saved-protocol fixtures, not original benchmark data or model evidence."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "saved_codejudge", ROOT / "scripts/verify_codejudge_saved.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture():
    row = {
        "input": "unchanged original fixture prompt",
        "answer": "C",
        "data_id": 1,
        "task_id": 77,
    }
    profile = {
        "model": "fixture-model",
        "max_output_tokens": 8192,
        "disable_thinking": True,
    }
    record = {
        "file": "CodeJudge_Eval_0shot_easy.json",
        "row_index": 0,
        "data_id": 1,
        "task_id": 77,
        "original_prompt_sha256": MODULE.digest(row["input"]),
        "reference_answer": "C",
        "request": MODULE.request_body(row, profile),
        "response": {
            "model": "fixture-model",
            "choices": [{"message": {"content": "C"}, "finish_reason": "stop"}],
        },
        "prediction": "C",
    }
    return row, profile, record


def test_saved_original_request_response_recomputed_without_network():
    row, profile, record = fixture()
    assert MODULE.verify_record(record, row, record["file"], 0, profile) == "C"


@pytest.mark.parametrize(
    "field,value",
    [
        ("reference_answer", "A"),
        ("prediction", "B"),
        ("data_id", 999),
        ("row_index", 1),
    ],
)
def test_changed_reference_extraction_or_original_identity_rejected(field, value):
    row, profile, record = fixture()
    record[field] = value
    with pytest.raises(ValueError):
        MODULE.verify_record(record, row, "CodeJudge_Eval_0shot_easy.json", 0, profile)


def test_prompt_rewrite_or_transport_failure_not_accepted():
    row, profile, original = fixture()
    record = deepcopy(original)
    record["request"]["messages"][0]["content"] += " rewritten fixture"
    with pytest.raises(ValueError, match="request"):
        MODULE.verify_record(record, row, record["file"], 0, profile)
    record = deepcopy(original)
    record["error_type"] = "fixture-transport-error"
    with pytest.raises(ValueError, match="transport"):
        MODULE.verify_record(record, row, record["file"], 0, profile)
