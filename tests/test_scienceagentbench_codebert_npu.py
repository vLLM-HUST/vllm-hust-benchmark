"""CPU binding fixtures, never represented as original neural scoring evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "science_codebert",
    Path(__file__).resolve().parents[1] / "scripts/scienceagentbench_codebert_npu.py",
)
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_binding_preserves_original_text_and_source_arguments():
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return ("precision", "recall", "f1")

    score = entry.bound_score(original, Path("/private/original-model"), "npu:0")
    payload = {
        "cands": ["unchanged prediction"],
        "refs": ["unchanged reference"],
        "lang": "python",
        "sources": ["unchanged task"],
    }
    assert score(**payload) == ("precision", "recall", "f1")
    assert calls == [
        {
            **payload,
            "model_type": str(Path("/private/original-model")),
            "num_layers": 11,
            "device": "npu:0",
        }
    ]
    assert set(payload) == {"cands", "refs", "lang", "sources"}


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "npu", "npu:-1", "npu:64"])
def test_no_implicit_cpu_or_cuda_fallback(device):
    with pytest.raises(ValueError, match="explicit logical NPU"):
        entry.bound_score(lambda **kwargs: None, "/model", device)


def test_no_silent_metric_parameter_or_language_changes():
    score = entry.bound_score(lambda **kwargs: None, "/model", "npu:0")
    with pytest.raises(ValueError, match="arguments"):
        score(cands=["p"], refs=["r"], lang="python", sources=["s"], idf=True)
    with pytest.raises(ValueError, match="Python"):
        score(cands=["p"], refs=["r"], lang="javascript", sources=["s"])


def test_reject_changed_source_before_compilation(tmp_path):
    p = tmp_path / "compute_scores.py"
    p.write_text("raise RuntimeError('must not execute')", encoding="utf-8")
    with pytest.raises(ValueError, match="pinned original"):
        entry.original_metric(p, lambda **kwargs: None)


def test_real_control_is_bound_to_executed_entry_not_native_delivery():
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs/workload-data"
    receipt_path = "docs/workload-data/validation/scienceagentbench-codebert-original-npu-control-20261101.json"
    receipt = json.loads((root / receipt_path).read_text(encoding="utf-8"))
    assert (
        receipt["public_entry_sha256"]
        == hashlib.sha256(
            (root / receipt["public_entry"]).read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
    )
    assert receipt["source_checkbox_eligible"] is False
    assert (
        receipt["native_scientific_tasks_completed"] == receipt["generator_calls"] == 0
    )
    assert receipt["completed_npu_forwards"] == 3
    assert receipt["independent_audit_model_calls"] == 0
    assert receipt["generated_code_executed"] is False
    assert receipt["native_exit_code"] == 0
    assert receipt["exclusive_lease_released"] is True
    assert receipt["protected_services_same_birth"] is True
    source = json.loads(
        (docs / "dataset-validation-status.json").read_text(encoding="utf-8")
    )["delivery_inventory"]["SCIENCEAGENTBENCH"]
    assert source["current_auxiliary_metric_control"] == receipt_path
    final = json.loads((root / source["delivery_receipt"]).read_text(encoding="utf-8"))
    assert source["checked"] is True and final["source_checkbox_eligible"] is True
    assert final["native_evidence"] != receipt_path
