"""Trusted-counter subprocess fixtures, not a QReCC collection or native run."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qrecc_index_probe", ROOT / "scripts/probe_qrecc_original_index.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    (index / "segments_fixture").write_bytes(b"not a real index")
    jar = tmp_path / "fixture.jar"
    jar.write_bytes(b"not a real JAR")
    monkeypatch.setattr(
        module, "JAR_SHA256", hashlib.sha256(jar.read_bytes()).hexdigest()
    )
    return dict(
        index=index,
        java="fixture-java",
        jar=jar,
        output=tmp_path / "probe",
        expected_documents=3,
    )


def responses(args, **kwargs):
    return SimpleNamespace(
        returncode=0,
        stdout="" if "-version" in args else "3\n",
        stderr='openjdk version "11.0.32.1"\n' if "-version" in args else "",
    )


def test_exact_count_is_not_native_delivery_or_corpus_provenance(inputs, monkeypatch):
    before = (inputs["index"] / "segments_fixture").read_bytes()
    monkeypatch.setattr(module.subprocess, "run", responses)
    report = module.probe(**inputs)
    assert report["observed_documents"] == report["expected_documents"] == 3
    assert (
        report["native_tasks_executed"] == 0 and not report["source_checkbox_eligible"]
    )
    assert not report["corpus_provenance_verified"]
    assert (inputs["index"] / "segments_fixture").read_bytes() == before
    code = (inputs["output"] / "CountDocuments.java").read_text()
    assert "DirectoryReader.open" in code and "IndexWriter" not in code


@pytest.mark.parametrize("kind", ["wrong-count", "error", "not-a-count"])
def test_failed_read_or_mismatched_actual_count_is_retained(inputs, monkeypatch, kind):
    def run(args, **kwargs):
        if "-version" in args:
            return responses(args, **kwargs)
        return SimpleNamespace(
            returncode=1 if kind == "error" else 0,
            stdout="4\n" if kind == "wrong-count" else "not a count",
            stderr="fixture failure",
        )

    monkeypatch.setattr(module.subprocess, "run", run)
    with pytest.raises(ValueError):
        module.probe(**inputs)
    receipt = json.loads((inputs["output"] / "receipt-private.json").read_text())
    assert (
        receipt["status"] == "FAILED_INDEX_PROBE_RETAINED"
        and not receipt["source_checkbox_eligible"]
    )
    assert (inputs["output"] / "counter-stderr.txt").read_text() == "fixture failure"


def test_changed_jar_rejected_before_java_or_output_creation(inputs, monkeypatch):
    inputs["jar"].write_bytes(b"changed")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: pytest.fail("must not execute changed software"),
    )
    with pytest.raises(ValueError, match="unchanged original"):
        module.probe(**inputs)
    assert not inputs["output"].exists()


def test_probe_output_cannot_contaminate_index(inputs):
    inputs["output"] = inputs["index"] / "new-output"
    with pytest.raises(ValueError, match="outside checkout and index"):
        module.probe(**inputs)
    assert not inputs["output"].exists()


def test_default_plan_never_touches_missing_index(tmp_path):
    path = tmp_path / "absent"
    out = subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts/probe_qrecc_original_index.py"),
            "--index",
            str(path),
            "--java",
            str(path),
            "--jar",
            str(path),
            "--output",
            str(path),
        ],
        text=True,
    )
    assert json.loads(out)["status"] == "PLAN_ONLY" and not path.exists()
