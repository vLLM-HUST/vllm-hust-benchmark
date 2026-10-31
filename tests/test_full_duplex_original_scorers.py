"""Invented transport fixtures; real publisher examples are server evidence."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "full_duplex_scorers", ROOT / "scripts/prepare_full_duplex_original_scorers.py"
)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def test_offline_plan_is_explicit_subset():
    result = subprocess.run(
        [sys.executable, str(SPEC.origin)], capture_output=True, text=True, check=True
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert len(plan["selected_files"]) == 11
    assert not any("node_modules" in name for name in plan["selected_files"])


def test_acquisition_and_read_only_reuse(tmp_path, monkeypatch):
    raw = b"invented publisher software fixture"
    identity = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    monkeypatch.setattr(ENTRY, "FILES", {"nested/example.json": identity})
    output = tmp_path / "external"
    result = ENTRY.acquire(output, opener=lambda *a, **k: io.BytesIO(raw))
    assert result["selected_original_files_verified"] == 1
    assert result["source_checkbox_eligible"] is False
    assert result["whole_repository_verified"] is False
    assert ENTRY.verify(output) == result
    with pytest.raises(FileExistsError):
        ENTRY.acquire(output)
    (output / "nested/example.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Git blob mismatch"):
        ENTRY.verify(output)


def test_corrupt_or_oversized_download_not_written(tmp_path, monkeypatch):
    monkeypatch.setattr(ENTRY, "FILES", {"bad.py": "0" * 40})
    output = tmp_path / "external"
    with pytest.raises(ValueError, match="Git blob mismatch"):
        ENTRY.acquire(output, opener=lambda *a, **k: io.BytesIO(b"wrong"))
    assert not (output / "bad.py").exists()
    with pytest.raises(ValueError):
        ENTRY.check_bytes("bad.py", b"x" * (ENTRY.MAX_FILE_BYTES + 1))


def test_repository_output_rejected_before_write():
    with pytest.raises(ValueError, match="external server storage"):
        ENTRY.acquire(ROOT / "work/forbidden-full-duplex-data")
