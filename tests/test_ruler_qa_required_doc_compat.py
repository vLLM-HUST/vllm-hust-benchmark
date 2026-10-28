"""Synthetic code fixture only; no publisher assets or model calls."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "ruler_qa_compat",
    Path(__file__).resolve().parents[1]
    / "scripts/prepare_ruler_qa_required_doc_compat.py",
)
compat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compat)
FIXTURE = """CALLS = []
def generate_input_output(index, num_docs):
    CALLS.append(num_docs)
    if num_docs < 10:
        raise ValueError('negative sample fixture')
    if num_docs > 10:
        raise AssertionError('overlength fixture')
    return (index, num_docs)

def generate_samples():
    index, used_docs, incremental = 0, 27, 10
    while True:
        try:
            return generate_input_output(index, used_docs)
        except:
            if used_docs > incremental:
                used_docs -= incremental
"""


def source_fixture(tmp_path, monkeypatch):
    source = tmp_path / "original"
    path = source / compat.TARGET
    path.parent.mkdir(parents=True)
    path.write_bytes(FIXTURE.encode())
    (source / "unchanged.txt").write_bytes(b"unchanged fixture")
    monkeypatch.setattr(
        compat, "ORIGINAL_QA_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return source


def test_floor_patch_keeps_original_source_and_renderer_and_all_required_docs(
    tmp_path, monkeypatch
):
    source = source_fixture(tmp_path, monkeypatch)
    output = tmp_path / "runtime"
    receipt = compat.compatibility_copy(source, output, allow_required_doc_floor=True)
    assert (source / compat.TARGET).read_bytes() == FIXTURE.encode()
    assert (output / "unchanged.txt").read_bytes() == b"unchanged fixture"
    assert receipt["question_renderer_ast_preserved"] and receipt["explicit_opt_in"]
    assert (
        not receipt["original_checkout_overwritten"]
        and not receipt["inference_or_scoring_executed"]
    )
    scope = {
        "QAS": [{"context": list(range(10))}],
        "args": SimpleNamespace(pre_samples=0),
    }
    exec(
        compile(
            (output / compat.TARGET).read_text(),
            "<synthetic-compatibility-fixture>",
            "exec",
        ),
        scope,
    )
    assert scope["generate_samples"]() == (0, 10)
    assert scope["CALLS"] == [27, 17, 10]
    assert receipt["runtime_source_sha256"] != receipt["original_source_sha256"]


def test_no_opt_in_means_no_runtime_copy(tmp_path, monkeypatch):
    source = source_fixture(tmp_path, monkeypatch)
    output = tmp_path / "runtime"
    with pytest.raises(ValueError, match="opt-in"):
        compat.compatibility_copy(source, output)
    assert not output.exists()


def test_unknown_source_never_patched(tmp_path, monkeypatch):
    source = source_fixture(tmp_path, monkeypatch)
    (source / compat.TARGET).write_bytes(b"modified fixture")
    with pytest.raises(ValueError, match="source mismatch"):
        compat.compatibility_copy(
            source, tmp_path / "runtime", allow_required_doc_floor=True
        )


def test_no_in_place_nested_or_overwriting_copy(tmp_path, monkeypatch):
    source = source_fixture(tmp_path, monkeypatch)
    for output in [source, source / "runtime", tmp_path]:
        with pytest.raises(ValueError, match="separate"):
            compat.compatibility_copy(source, output, allow_required_doc_floor=True)
    output = tmp_path / "already-exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        compat.compatibility_copy(source, output, allow_required_doc_floor=True)
