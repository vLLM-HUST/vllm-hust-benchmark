"""No neural execution; test explicit device and checkpoint boundary handling."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/run_full_duplex_original_asr_npu.py"
)
SPEC = importlib.util.spec_from_file_location("duplex_npu_asr", SCRIPT)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


@pytest.mark.parametrize("version", ["v1.0", "v1.5"])
def test_plan_needs_no_model_files_or_neural_environment(tmp_path, version):
    command = [sys.executable, str(SCRIPT), "--version", version]
    for flag in ["omni-root", "reference-source", "tree", "nemo", "output"]:
        command.extend(["--" + flag, str(tmp_path / flag)])
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY_NO_MODEL_CALLS"
    assert plan["source_checkbox_eligible"] is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("devices", [[], ["cpu"], ["cuda:0"], ["npu:0", "cpu"]])
def test_neural_cpu_or_mixed_parameters_are_not_accepted(devices):
    model = argparse.Namespace(
        parameters=lambda: [argparse.Namespace(device=x) for x in devices]
    )
    with pytest.raises(ValueError, match="all neural parameters"):
        ENTRY.require_npu_parameters(model)


def test_npu_parameters_are_explicit():
    model = argparse.Namespace(parameters=lambda: [argparse.Namespace(device="npu:0")])
    assert ENTRY.require_npu_parameters(model) == ["npu:0"]


def test_wrong_checkpoint_fails_before_runtime_imports(tmp_path):
    path = tmp_path / "not-a-checkpoint"
    path.write_bytes(b"invented fixture")
    with pytest.raises(ValueError, match="checkpoint differs"):
        ENTRY.verify_checkpoint(path)


@pytest.mark.parametrize("version", ["v1.0", "v1.5"])
@pytest.mark.parametrize("status", ["valid", "invalid", "empty"])
def test_original_execution_delegation_preserves_full_protocol_and_rejects_failed_counts(
    tmp_path, monkeypatch, version, status
):
    # Inert orchestration, not neural evidence: completed native runs are separate.
    args = argparse.Namespace(
        version=version,
        tree=tmp_path / "tree",
        reference_source=tmp_path / "reference",
        nemo=tmp_path / "original.nemo",
        omni_root=tmp_path / "omni",
        output=tmp_path / "output",
        engine="minicpmo-npu-derived",
    )
    calls = []
    model = argparse.Namespace(parameters=lambda: [argparse.Namespace(device="npu:0")])
    counts = (
        {name: 3 for name in ENTRY.V10_ASR_SUBSETS} if version == "v1.0" else {"ok": 48}
    )
    if status == "invalid":
        counts = {"invalid": 1}
    elif status == "empty":
        counts = {}

    def load_model(path, device):
        assert path == args.nemo and device == "npu:0"
        calls.append("load_original_model")
        return model

    def transcribe(tree, paths, checkpoint, sha, device, **kwargs):
        assert (tree, paths, checkpoint, sha, device) == (
            args.tree,
            "verified",
            args.nemo,
            ENTRY.CHECKPOINT_SHA256,
            "npu:0",
        )
        assert kwargs == {"model": model}
        calls.append("original_v10_transcribe")
        return counts

    def run_asr(native, engines, paths, hashes, **kwargs):
        assert native.phase == "asr" and native.device == "npu:0"
        assert native.tree == [args.engine + "=" + str(args.tree)]
        assert (
            native.manifest == []
            and native.manifest_projection is None
            and native.only == []
        )
        assert native.limit is None and native.retry_failed is False
        assert (
            native.nemo == args.nemo and native.nemo_sha256 == ENTRY.CHECKPOINT_SHA256
        )
        assert native.reference_source == args.reference_source
        assert kwargs == {"model": model}
        calls.append("original_v15_run_asr")
        return counts

    modules = {
        "torch": types.SimpleNamespace(
            npu=types.SimpleNamespace(
                set_device=lambda device: calls.append(("device", device))
            )
        ),
        "torch_npu": types.ModuleType("torch_npu"),
        "benchmarks": types.ModuleType("benchmarks"),
        "benchmarks.duplex": types.ModuleType("duplex"),
        "benchmarks.eval": types.ModuleType("eval"),
        "benchmarks.duplex.reference_source": types.SimpleNamespace(
            verify_reference=lambda *a: "verified"
        ),
        "benchmarks.duplex.reference_asr": types.SimpleNamespace(
            load_nemo_model=load_model, run_asr=run_asr
        ),
        "benchmarks.duplex.reference_v10": types.SimpleNamespace(transcribe=transcribe),
        "benchmarks.duplex.reference_core": types.SimpleNamespace(
            V10_REFERENCE_FILES="original-v10-files",
            HashCache=lambda path: types.SimpleNamespace(save=lambda: None),
        ),
        "benchmarks.eval.benchmark_duplex_reference": types.SimpleNamespace(
            open_engines=lambda native: []
        ),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(
        ENTRY,
        "verify_checkpoint",
        lambda path: calls.append("verify_original_checkpoint"),
    )
    if status == "valid":
        ENTRY.execute(args)
    else:
        with pytest.raises(RuntimeError, match="reported failed"):
            ENTRY.execute(args)
    assert calls == [
        "verify_original_checkpoint",
        ("device", 0),
        "load_original_model",
        "original_v10_transcribe" if version == "v1.0" else "original_v15_run_asr",
    ]
    receipt = json.loads((args.output / "qualification-public.json").read_text())
    assert receipt["status"] == (
        "ORIGINAL_ASR_COMPLETED_NOT_WHOLE_BENCHMARK" if status == "valid" else "FAILED"
    )
