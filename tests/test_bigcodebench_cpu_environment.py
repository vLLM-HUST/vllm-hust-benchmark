"""Tiny CPU/software fixtures only; never build an image or execute a task."""

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = load("verify_bigcodebench_cpu_environment")
context = load("prepare_bigcodebench_cpu_context")


def test_lock_is_exact_complete_cpu_environment():
    lock = ROOT / "configs/workload-native/bigcodebench-cpu-py310-aarch64.lock.txt"
    entries = {
        line.split("==")[0]: line.split("==")[1]
        for line in lock.read_text().splitlines()
        if line and not line.startswith("#")
    }
    versions = verifier.verify_locked_versions(lock, entries.__getitem__)
    assert len(versions) == len(entries) > 200
    assert versions["torch"] == "2.2.2"
    assert versions["scikit-image"] == "0.18.0"
    assert versions["cryptography"] == "38.0.0"
    assert versions["protobuf"] == "3.19.6"
    assert versions["e2b"] == "0.17.0"
    assert not any(name.startswith("nvidia-") for name in versions)


@pytest.mark.parametrize(
    "line",
    [
        "numpy>=1",
        "--index-url https://example.invalid",
        "name @ https://example.invalid",
    ],
)
def test_nonexact_or_external_lock_entries_rejected(tmp_path, line):
    lock = tmp_path / "lock.txt"
    lock.write_text(line)
    with pytest.raises(ValueError, match="exact named"):
        verifier.verify_locked_versions(lock, lambda name: "1")


def test_lock_rejects_drift_and_canonical_duplicate(tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("test_pkg==1\ntest-pkg==1\n")
    with pytest.raises(ValueError, match="duplicate"):
        verifier.verify_locked_versions(lock, lambda name: "1.0")
    lock.write_text("test_pkg==1\n")
    with pytest.raises(ValueError, match="mismatch"):
        verifier.verify_locked_versions(lock, lambda name: "2")


def test_context_copies_only_verified_code_and_explicit_runtime_files(
    tmp_path, monkeypatch
):
    source = tmp_path / "author"
    source.mkdir()
    (source / "selected.py").write_bytes(b"# inert original software fixture\n")
    (source / "PRIVATE_DO_NOT_COPY.jsonl").write_text("private fixture")
    (source / ".git").mkdir()
    monkeypatch.setattr(
        context,
        "verify_source",
        lambda path: {
            "selected.py": hashlib.sha256(
                (source / "selected.py").read_bytes()
            ).hexdigest()
        },
    )
    output = tmp_path / "context"
    result = context.prepare(source, output)
    assert result["original_source_files_verified"] == 1
    assert not result["raw_tasks_or_model_weights_copied"]
    assert sorted(path.name for path in (output / "author").iterdir()) == [
        "selected.py"
    ]
    assert (output / "Dockerfile").is_file()
    with pytest.raises(ValueError, match="fresh external"):
        context.prepare(source, output)


def test_context_rejects_overlapping_directory_before_source_read(
    tmp_path, monkeypatch
):
    source = tmp_path / "author"
    source.mkdir()
    monkeypatch.setattr(
        context, "verify_source", lambda path: pytest.fail("must reject first")
    )
    with pytest.raises(ValueError, match="fresh external"):
        context.prepare(source, source / "output")


def test_recipe_keeps_original_pins_and_no_runtime_thread_override():
    recipe = (ROOT / "configs/workload-native/bigcodebench-cpu.Dockerfile").read_text()
    assert "--no-build-isolation --no-deps scikit-image==0.18.0" in recipe
    assert "-c /author/Requirements/requirements-eval.txt" in recipe
    assert "verify_bigcodebench_cpu_environment.py" in recipe
    assert not any(
        "OMP_NUM_THREADS" in line
        for line in recipe.splitlines()
        if line.startswith("ENV ")
    )
    assert "COPY . " not in recipe
