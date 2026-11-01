"""Scalar transport/UID software fixtures; not original scientific execution."""

import hashlib
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "sab_isolated",
    Path(__file__).resolve().parents[1]
    / "scripts/scienceagentbench_isolated_scoring.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def fixture_receipt(pred, gold):
    return {
        "status": "ORIGINAL_AUXILIARY_METRIC_COMPLETED_NOT_SCIENTIFIC_TASK_DELIVERY",
        "model": "neulab/codebert-python",
        "model_layers": 11,
        "distribution_version": "0.4.1",
        "device": "npu:0",
        "completed_npu_forwards": 1,
        "results": [
            {
                "task_sha256": hashlib.sha256(b"original task").hexdigest(),
                "prediction_sha256": entry.digest(pred),
                "reference_sha256": entry.digest(gold),
                "codebert_score": 0.75,
            }
        ],
    }


def test_transport_uses_exact_actual_task_and_files_not_fallback(tmp_path):
    pred, gold = tmp_path / "prediction.py", tmp_path / "reference.py"
    pred.write_text("original prediction")
    gold.write_text("original reference")
    call = entry.verified_npu_scalar(fixture_receipt(pred, gold))
    assert call("original task", pred, gold) == 0.75
    with pytest.raises(ValueError, match="do not match"):
        call("different task", pred, gold)
    pred.write_text("changed prediction")
    with pytest.raises(ValueError, match="do not match"):
        call("original task", pred, gold)


@pytest.mark.parametrize(
    "key,value",
    [
        ("device", "cpu"),
        ("model_layers", 10),
        ("completed_npu_forwards", 0),
        ("distribution_version", "other"),
    ],
)
def test_no_unqualified_metric_receipt(tmp_path, key, value):
    p = tmp_path / "program.py"
    p.write_text("code")
    receipt = fixture_receipt(p, p)
    receipt[key] = value
    with pytest.raises(ValueError, match="actual original NPU"):
        entry.verified_npu_scalar(receipt)


def test_unprivileged_execution_preserves_original_timeout_and_command():
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return "saved original result"

    expected = ["python", "-m", "program_to_eval.pred_original"]
    call = entry.restricted_subprocess(run, expected, 1000, 1000)
    assert call(expected, capture_output=True, timeout=900) == "saved original result"
    command, kwargs = calls[0]
    assert command == expected and kwargs["timeout"] == 900
    assert kwargs["user"] == kwargs["group"] == 1000
    assert kwargs["extra_groups"] == []
    assert kwargs["env"]["HOME"] == "/home/nonroot"
    with pytest.raises(ValueError):
        call(expected, capture_output=True, timeout=30)
    with pytest.raises(ValueError):
        call(["python", "dangerous.py"], capture_output=True, timeout=900)


def test_root_scientific_execution_rejected():
    with pytest.raises(ValueError, match="non-root"):
        entry.restricted_subprocess(lambda *args, **kwargs: None, [], 0, 0)
