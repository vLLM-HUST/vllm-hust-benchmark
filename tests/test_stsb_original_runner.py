"""CPU protocol fixtures, not STS-B native model evidence."""

import hashlib
import ast
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "stsb_runner", Path(__file__).resolve().parents[1] / "scripts/run_stsb_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_fixed_full_dev_test_protocol_and_model():
    pins = json.loads(runner.PINS.read_text(encoding="utf-8"))
    assert pins["dataset"]["rows"] == 8628
    assert pins["protocol"]["expected_rows"] == {"dev": 1500, "test": 1379}
    assert pins["protocol"]["batch_size"] == 16
    assert pins["model"]["repo_id"] == "sentence-transformers/all-mpnet-base-v2"
    assert len(pins["model"]["files"]) == 13
    assert pins["framework_version"] == "5.1.2"
    assert len(pins["installed_source_files"]) == 6
    assert all(
        row["normalized_only_line_endings_equals_git_source"]
        for row in pins["installed_source_files"]
    )
    assert (
        pins["publisher_wheel"]["filename"]
        == "sentence_transformers-5.1.2-py3-none-any.whl"
    )


def test_validation_wrapper_does_not_change_original_sbert_model_type():
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    wrappers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "NativeEncoder"
    ]
    assert len(wrappers) == 1 and not wrappers[0].bases
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert any(node.func.id == "SentenceTransformer" for node in calls)


def test_encode_identity_preserves_order_and_original_whitespace():
    assert runner.input_identity(["a", "b"]) != runner.input_identity(["b", "a"])
    assert runner.input_identity([" a "]) != runner.input_identity(["a"])


def test_cached_embeddings_require_same_inputs_options_and_file(tmp_path):
    path = tmp_path / "embeddings.private.npy"
    path.write_bytes(b"fixture-only")
    options = {"normalize_embeddings": False, "precision": None, "batch_size": 16}
    call = {
        "input_sha256": runner.input_identity(["original"]),
        "encode_options": options,
        "file": path.name,
        "sha256": runner.sha(path),
    }
    assert (
        runner.validate_cached_call(
            call, sentences=["original"], kwargs=options, directory=tmp_path
        )
        == path
    )
    for sentences, kwargs in [
        (["rewritten"], options),
        (["original"], {**options, "normalize_embeddings": True}),
    ]:
        with pytest.raises(ValueError, match="input or protocol"):
            runner.validate_cached_call(
                call, sentences=sentences, kwargs=kwargs, directory=tmp_path
            )
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity mismatch"):
        runner.validate_cached_call(
            call, sentences=["original"], kwargs=options, directory=tmp_path
        )


def test_primary_git_file_verification_and_containment(tmp_path):
    path = tmp_path / "config.json"
    raw = b"{}\n"
    path.write_bytes(raw)
    pinned = {
        "path": path.name,
        "bytes": len(raw),
        "git_blob": hashlib.sha1(b"blob 3\0" + raw).hexdigest(),
    }
    runner.verify_files(tmp_path, [pinned])
    with pytest.raises(ValueError, match="Git object mismatch"):
        runner.verify_files(tmp_path, [{**pinned, "git_blob": "0" * 40}])
    with pytest.raises(ValueError, match="path or size mismatch"):
        runner.verify_files(tmp_path, [{**pinned, "path": "../escape.json"}])


def test_no_cpu_model_no_training_split_no_implicit_cache():
    base = ["--data-root", "data", "--output", "out"]
    for extra in [
        ["--model-root", "model", "--device", "cpu"],
        ["--mode", "score"],
        ["--model-root", "model", "--reuse", "prior"],
        ["--model-root", "model", "--split", "train"],
    ]:
        with pytest.raises(SystemExit):
            runner.parse_args(base + extra)
    assert (
        runner.parse_args(base + ["--model-root", "model", "--device", "npu:1"]).mode
        == "run"
    )
    assert (
        runner.parse_args(base + ["--mode", "score", "--reuse", "prior"]).mode
        == "score"
    )
