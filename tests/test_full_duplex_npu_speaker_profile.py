"""Inert artifact/profile fixtures; no ACL runtime or neural model execution."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "duplex_npu_speaker",
    Path(__file__).resolve().parents[1] / "scripts/full_duplex_npu_speaker.py",
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def fixture(tmp_path, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "speaker.om").write_bytes(b"inert-not-a-model")
    graph = root / "derived.onnx"
    graph.write_bytes(b"inert-not-an-onnx-graph")
    monkeypatch.setattr(ENTRY, "DERIVED_GRAPH_SHA256", ENTRY.digest(graph))
    receipt = {
        "passed": True,
        "neural_execution": True,
        "execution_device": "NPU",
        "input_shape": [1, 1682, 80],
        "output_shape": [1, 192],
        "input_features_sha256": ENTRY.INPUT_SHA256,
        "output_sha256": ENTRY.OUTPUT_SHA256,
        "model_om_sha256": hashlib.sha256(b"inert-not-a-model").hexdigest(),
    }
    (root / "qualification.json").write_text(json.dumps(receipt))
    profile = {
        "schema_version": "full-duplex-npu-speaker-profile/v1",
        "model_om": "speaker.om",
        "derived_graph": "derived.onnx",
        "qualification": "qualification.json",
    }
    path = root / "profile.json"
    path.write_text(json.dumps(profile))
    return path, receipt


def test_bundle_paths_are_relocatable_and_byte_bound(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    files, receipt = ENTRY.load_profile(path)
    assert files["model_om"] == path.parent / "speaker.om"
    assert receipt["input_shape"] == [1, 1682, 80]
    files["model_om"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash differs"):
        ENTRY.load_profile(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("execution_device", "CPU"),
        ("neural_execution", False),
        ("input_shape", [1, 10, 80]),
        ("output_sha256", "not-qualified"),
    ],
)
def test_failed_cpu_or_other_reference_qualification_is_refused(
    tmp_path, monkeypatch, field, value
):
    path, receipt = fixture(tmp_path, monkeypatch)
    receipt[field] = value
    (path.parent / "qualification.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="qualification or artifact hash differs"):
        ENTRY.load_profile(path)


def test_profile_cannot_escape_its_bundle(tmp_path, monkeypatch):
    path, _ = fixture(tmp_path, monkeypatch)
    (tmp_path / "outside.onnx").write_bytes(b"inert")
    profile = json.loads(path.read_text())
    profile["derived_graph"] = "../outside.onnx"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="escapes"):
        ENTRY.load_profile(path)


def test_missing_explicit_profile_fails_before_runtime_import(monkeypatch):
    monkeypatch.delenv("FULL_DUPLEX_NPU_SPEAKER_PROFILE", raising=False)
    with pytest.raises(ValueError, match="must name a qualified bundle"):
        ENTRY.NpuSpeakerSession("not-read-without-profile.onnx")


@pytest.mark.parametrize("failure", [None, "execute", "output", "cleanup"])
def test_session_releases_every_temporary_resource_on_success_or_failure(
    monkeypatch, failure
):
    data = np.zeros((1, 1682, 80), dtype=np.float32)
    expected = np.arange(192, dtype=np.float32).reshape(1, 192)
    monkeypatch.setattr(
        ENTRY, "INPUT_SHA256", hashlib.sha256(data.tobytes()).hexdigest()
    )
    monkeypatch.setattr(
        ENTRY, "OUTPUT_SHA256", hashlib.sha256(expected.tobytes()).hexdigest()
    )
    released = []
    counters = {"pointer": 10, "dataset": 100, "buffer": 200}

    def allocate(size, flag):
        counters["pointer"] += 1
        return counters["pointer"], 0

    def handle(kind):
        counters[kind] += 1
        return counters[kind]

    def memcpy(destination, capacity, source, size, direction):
        if direction == 2:
            destination[:] = expected + (1 if failure == "output" else 0)
        return 0

    def release(kind, value):
        released.append((kind, value))
        return 1 if failure == "cleanup" and kind == "buffer" else 0

    session = ENTRY.NpuSpeakerSession.__new__(ENTRY.NpuSpeakerSession)
    session.device, session.model_id = 0, 1
    session.np = np
    session.torch = SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 0))
    session.input_size, session.output_size = data.nbytes, expected.nbytes
    session.output_shape = [1, 192]
    session.acl = SimpleNamespace(
        rt=SimpleNamespace(
            malloc=allocate, memcpy=memcpy, free=lambda value: release("pointer", value)
        ),
        util=SimpleNamespace(numpy_to_ptr=lambda array: array),
        create_data_buffer=lambda pointer, size: handle("buffer"),
        destroy_data_buffer=lambda value: release("buffer", value),
        mdl=SimpleNamespace(
            create_dataset=lambda: handle("dataset"),
            add_dataset_buffer=lambda dataset, buffer: (dataset, 0),
            execute=lambda *args: 1 if failure == "execute" else 0,
            destroy_dataset=lambda value: release("dataset", value),
        ),
    )
    if failure:
        with pytest.raises((ValueError, RuntimeError)):
            session.run(None, {"input": data})
    else:
        np.testing.assert_array_equal(session.run(None, {"input": data})[0], expected)
    assert len(released) == 6
    assert sum(kind == "pointer" for kind, _ in released) == 2
