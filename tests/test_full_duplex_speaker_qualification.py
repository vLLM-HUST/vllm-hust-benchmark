import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "duplex_speaker_qualification", ROOT / "scripts/qualify_full_duplex_npu_speaker.py"
)
ENTRY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ENTRY)


def test_default_plan_does_not_read_models_or_use_device(tmp_path):
    args = [sys.executable, str(ROOT / "scripts/qualify_full_duplex_npu_speaker.py")]
    for name in [
        "original-model",
        "derived-graph",
        "model-om",
        "reference-audio",
        "output",
    ]:
        args.extend(["--" + name, str(tmp_path / name)])
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["status"] == "PLAN_ONLY_NO_DEVICE_USE"
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("visible", [None, "", "0,1", "auto"])
def test_explicit_one_allocated_device_required_before_output_creation(
    tmp_path, monkeypatch, visible
):
    if visible is None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="single allocated"):
        ENTRY.qualify(argparse.Namespace(output=tmp_path / "output"))
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("failure", [None, "run", "run-and-cleanup"])
def test_receipt_is_written_only_as_passed_after_successful_control_and_release(
    tmp_path, monkeypatch, failure
):
    # Inert orchestration fixture; no Torch, ACL or real neural network runs.
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1")
    original, graph, model = [
        tmp_path / name for name in ["original.onnx", "derived.onnx", "model.om"]
    ]
    for path in [original, graph, model]:
        path.write_bytes(b"inert fixture")
    monkeypatch.setattr(ENTRY, "ORIGINAL_ONNX_SHA256", ENTRY.digest(original))
    monkeypatch.setattr(ENTRY, "DERIVED_GRAPH_SHA256", ENTRY.digest(graph))
    monkeypatch.setattr(
        ENTRY,
        "reference_features",
        lambda path: np.zeros((1, 1682, 80), dtype=np.float32),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(set_device=lambda device: None)),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    calls = []

    class Session:
        def run(self, names, inputs):
            calls.append("run")
            if failure:
                raise ValueError("primary control failure")
            return [np.ones((1, 192), dtype=np.float32)]

        def close(self):
            calls.append("close")
            if failure == "run-and-cleanup":
                raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(
        ENTRY,
        "NpuSpeakerSession",
        SimpleNamespace(for_reference_qualification=lambda *args: Session()),
    )
    args = argparse.Namespace(
        original_model=original,
        derived_graph=graph,
        model_om=model,
        reference_audio=tmp_path / "reference.wav",
        output=tmp_path / "bundle",
    )
    if failure:
        with pytest.raises(ValueError, match="primary control failure") as captured:
            ENTRY.qualify(args)
        if failure == "run-and-cleanup":
            assert any("secondary cleanup" in note for note in captured.value.__notes__)
        assert not (args.output / "profile.json").exists()
    else:
        ENTRY.qualify(args)
        assert (args.output / "profile.json").is_file()
    receipt = json.loads((args.output / "qualification-public.json").read_text())
    assert receipt["passed"] is (failure is None)
    assert calls == ["run", "close"]
