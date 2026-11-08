import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "speaker_graph_entry", SCRIPTS / "prepare_full_duplex_speaker_graph.py"
)
ENTRY = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ENTRY)


def test_default_plan_never_reads_model_or_creates_output(tmp_path):
    output = tmp_path / "not-created"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_full_duplex_speaker_graph.py"),
            "--original",
            str(tmp_path / "missing.onnx"),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "PLAN_ONLY"
    assert report["neural_execution"] is False
    assert not output.exists()


def test_wrong_original_hash_is_rejected_before_onnx_import(tmp_path):
    original = tmp_path / "bad.onnx"
    original.write_bytes(b"not the published graph")
    with pytest.raises(ValueError, match="hash differs"):
        ENTRY.prepare(original, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("length", [1, 41, 99, 100, 101, 200, 841])
@pytest.mark.parametrize("opset", [17, 18])
def test_pool_lowering_preserves_partial_window_semantics(length, opset):
    onnx = pytest.importorskip("onnx")
    import numpy as np
    from onnx import helper
    from onnx.reference import ReferenceEvaluator

    pool = helper.make_node(
        "AveragePool",
        ["x"],
        ["y"],
        name="pool",
        ceil_mode=1,
        kernel_shape=[100],
        pads=[0, 0],
        strides=[100],
    )
    nodes, values = ENTRY.lower_pool(pool, [1, 3, length], opset)
    model = helper.make_model(
        helper.make_graph(
            nodes,
            "isolated_pool_primitive",
            [
                helper.make_tensor_value_info(
                    "x", onnx.TensorProto.FLOAT, [1, 3, length]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "y", onnx.TensorProto.FLOAT, [1, 3, (length + 99) // 100]
                )
            ],
            initializer=values,
        ),
        opset_imports=[helper.make_opsetid("", opset)],
    )
    onnx.checker.check_model(model)
    sample = (
        np.random.default_rng(20261106)
        .standard_normal([1, 3, length])
        .astype(np.float32)
    )
    actual = ReferenceEvaluator(model).run(None, {"x": sample})[0]
    expected = np.stack(
        [sample[:, :, i : i + 100].mean(axis=2) for i in range(0, length, 100)], axis=2
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_rejects_other_pooling_contract():
    onnx = pytest.importorskip("onnx")
    node = onnx.helper.make_node("AveragePool", ["x"], ["y"], kernel_shape=[3])
    with pytest.raises(ValueError, match="original disjoint"):
        ENTRY.lower_pool(node, [1, 3, 101], 17)


def test_compile_requires_explicit_execution(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "prepare_full_duplex_speaker_graph.py"),
            "--original",
            str(tmp_path / "missing.onnx"),
            "--output",
            str(tmp_path / "out"),
            "--compile",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--compile requires --execute" in result.stderr
    assert not (tmp_path / "out").exists()


def test_missing_compiler_retains_failure_receipt(tmp_path, monkeypatch):
    output = tmp_path / "output"

    def fake_prepare(original, destination):
        destination.mkdir()
        graph = destination / "fixture.onnx"
        graph.write_bytes(b"non-neural orchestration fixture")
        return graph

    monkeypatch.setattr(ENTRY, "prepare", fake_prepare)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "entry",
            "--original",
            str(tmp_path / "original.onnx"),
            "--output",
            str(output),
            "--execute",
            "--compile",
            "--atc",
            str(tmp_path / "missing-atc"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        ENTRY.main()
    receipt = json.loads((output / "qualification-public.json").read_text())
    assert receipt["status"] == "COMPILATION_FAILED"
    assert receipt["neural_execution"] is False
