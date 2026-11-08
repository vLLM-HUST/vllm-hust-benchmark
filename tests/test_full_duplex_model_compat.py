"""Invented source fixtures; no upstream framework imports or neural execution."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "duplex_model_compat", ROOT / "scripts/prepare_full_duplex_model_compat.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def fixture(tmp_path, monkeypatch):
    source = tmp_path / "upstream"
    (source / "benchmarks").mkdir(parents=True)
    (source / "benchmarks/untouched.py").write_text("# invented fixture\nvalue = 1\n")
    specs = deepcopy(ENTRY.SPECS)
    for name, spec in specs.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("vocoder.py"):
            text = "class Fixture:\n    def f(self, model_path, source):\n"
            text += (
                spec["replacements"][0][0] + "\n" + spec["replacements"][1][0] + "\n"
            )
        elif name.endswith("code2wav.py"):
            text = (
                "class Fixture:\n    def f(self, resolved_device, device):\n"
                '        if resolved_device.type not in {"cuda", "xpu"}:\n'
                '            raise ValueError(f"Token2wav requires a CUDA or XPU device, got {device}")\n'
            )
        else:
            text = (
                "def f(obj):\n    if isinstance(obj, Tensor):\n"
                '        if obj.device.type in ("cpu", "musa"):\n'
                "            return {obj.device.type}\n"
            )
        path.write_text(text)
        spec["sha256"] = ENTRY.sha(path)
    monkeypatch.setattr(ENTRY, "SPECS", specs)
    return source


def test_preparation_copies_sources_without_changing_upstream(tmp_path, monkeypatch):
    source = fixture(tmp_path, monkeypatch)
    files, _ = ENTRY.inventory(source)
    before = {name: (source / name).read_bytes() for name in files}
    output = tmp_path / "compatibility"
    report = ENTRY.prepare(source, output, files)
    assert report["status"] == "SOURCE_COMPATIBILITY_COPY_PREPARED_NOT_EXECUTED"
    assert report["neural_execution"] is False
    assert len(report["changed_files"]) == 3
    assert {name: (source / name).read_bytes() for name in files} == before
    assert (output / "source/benchmarks/untouched.py").read_bytes() == before[
        "benchmarks/untouched.py"
    ]
    assert (output / "runtime-helpers/full_duplex_npu_speaker.py").is_file()
    vocoder = (output / "source" / (ENTRY.PREFIX + "token2wav/vocoder.py")).read_text()
    assert "CPUExecutionProvider" not in vocoder
    assert "from full_duplex_reference_audio import load_reference_wave" in vocoder


def test_already_patched_or_changed_source_is_refused(tmp_path, monkeypatch):
    source = fixture(tmp_path, monkeypatch)
    path = source / next(iter(ENTRY.SPECS))
    path.write_text(path.read_text() + "# altered\n")
    with pytest.raises(ValueError, match="pinned original affected file differs"):
        ENTRY.inventory(source)


def test_output_must_not_overlap_original_source(tmp_path, monkeypatch):
    source = fixture(tmp_path, monkeypatch)
    files, _ = ENTRY.inventory(source)
    with pytest.raises(ValueError, match="separate from upstream"):
        ENTRY.prepare(source, source / "patched", files)


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    source = fixture(tmp_path, monkeypatch)
    files, _ = ENTRY.inventory(source)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        ENTRY.prepare(source, output, files)
    assert list(output.iterdir()) == []


def test_failed_copy_retains_failure_receipt(tmp_path, monkeypatch):
    source = fixture(tmp_path, monkeypatch)
    files, _ = ENTRY.inventory(source)
    (source / "benchmarks/untouched.py").write_text("# changed after inventory\n")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="source changed"):
        ENTRY.prepare(source, output, files)
    receipt = json.loads((output / "qualification-public.json").read_text())
    assert receipt["status"] == "FAILED"
    assert receipt["source_checkbox_eligible"] is False


def test_startup_configuration_keeps_declared_native_profile():
    profile = yaml.safe_load(
        (ROOT / "configs/workload-native/full-duplex-minicpmo-npu.yaml").read_text()
    )
    assert profile["config_cls"] == "MiniCPMODuplexPipelineConfig"
    assert profile["model_path"] == "/models"
    assert profile["reference_audio"] == "/models/assets/system_ref_audio.wav"
    assert profile["max_sessions"] == 1
    assert profile["stages"] == {
        stage: {"engine": {"attention_backend": "ascend", "disable_cuda_graph": True}}
        for stage in ["thinker", "talker"]
    }
