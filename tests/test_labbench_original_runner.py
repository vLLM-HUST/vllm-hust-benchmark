"""Model-free fixtures; not publisher task or NPU execution evidence."""

import hashlib
import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "labbench_runner", ROOT / "scripts/run_labbench_original.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://remote.example/v1",
        "http://user:secret@localhost/v1",  # pragma: allowlist secret - deliberately rejected dummy fixture
        "http://localhost/v1?key=x",
    ],
)
def test_reject_external_or_credentialed_endpoint(endpoint):
    with pytest.raises(ValueError):
        RUNNER.local_endpoint(endpoint)


def test_fixed_original_order_per_subset():
    originals = [("a", object()), ("b", object()), ("a", object()), ("b", object())]
    assert RUNNER.select_instances(originals, 1) == originals[:2]
    assert RUNNER.select_instances(originals, None) == originals
    with pytest.raises(ValueError):
        RUNNER.select_instances(originals, 0)


def test_verified_archive_and_unchanged_source_required(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "task.py").write_bytes(b"# original fixture\n")
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.write(source / "task.py", "publisher/task.py")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (
        RUNNER.verify_source(source, archive, expected_sha=sha, expected_files=1) == 1
    )
    (source / "task.py").write_bytes(b"# modified\n")
    with pytest.raises(ValueError, match="modified"):
        RUNNER.verify_source(source, archive, expected_sha=sha, expected_files=1)


def test_unregistered_executable_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "task.py").write_bytes(b"# original fixture\n")
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.write(source / "task.py", "publisher/task.py")
    (source / "injected.py").write_bytes(b"# unexpected\n")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="unexpected"):
        RUNNER.verify_source(source, archive, expected_sha=sha, expected_files=1)
