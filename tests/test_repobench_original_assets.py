"""Small source fixtures only, not publisher acquisition or NPU evidence."""

import hashlib
import importlib.util
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "repobench_original_assets", ROOT / "scripts/prepare_repobench_original_assets.py"
)
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


def test_default_plan_does_not_create_assets(tmp_path):
    output = tmp_path / "fresh"
    result = PREP.prepare(output)
    assert result["status"] == "PLAN_ONLY"
    assert not result["source_checkbox_eligible"]
    assert not output.exists()


def test_original_archive_reuse_and_readonly_verification(tmp_path, monkeypatch):
    archive = tmp_path / "original.zip"
    payload = b"# unchanged original fixture\n"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("publisher-fixed/run.py", payload)
    monkeypatch.setattr(PREP, "ARCHIVE_BYTES", archive.stat().st_size)
    monkeypatch.setattr(
        PREP, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(PREP, "SOURCE_BYTES", len(payload))
    monkeypatch.setattr(PREP, "SOURCE_FILES", 1)
    output = tmp_path / "assets"
    result = PREP.prepare(output, execute=True, reuse_archive=archive)
    assert result["verified_files"] == 1
    assert (output / "source/run.py").read_bytes() == payload
    before = {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}
    assert PREP.prepare(output, verify_only=True)["verified_files"] == 1
    assert before == {p: p.stat().st_mtime_ns for p in output.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="fresh output"):
        PREP.prepare(output, execute=True)
    (output / "source/run.py").write_bytes(b"modified source")
    with pytest.raises(ValueError, match="missing or modified"):
        PREP.prepare(output, verify_only=True)


def test_bad_reused_archive_does_not_create_output(tmp_path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"bad original")
    output = tmp_path / "untouched"
    with pytest.raises(ValueError, match="size mismatch"):
        PREP.prepare(output, execute=True, reuse_archive=archive)
    assert not output.exists()
