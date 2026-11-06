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


def test_optional_client_plan_adds_only_four_pinned_software_files():
    result = subprocess.run(
        [sys.executable, str(SPEC.origin), "--include-client"],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert plan["status"] == "PLAN_ONLY_NO_DOWNLOAD"
    assert plan["client_sources_included"]
    assert len(plan["selected_files"]) == 15
    assert set(plan["selected_files"]) == set(ENTRY.FILES) | set(ENTRY.CLIENT_FILES)
    assert plan["model_calls"] == 0


def test_full_reference_plan_covers_both_versions_without_execution():
    result = subprocess.run(
        [sys.executable, str(SPEC.origin), "--include-reference"],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert len(plan["selected_files"]) == 22
    assert plan["client_sources_included"]
    assert plan["complete_reference_sources_included"]
    assert plan["model_calls"] == 0
    assert set(plan["selected_files"]) == set(
        ENTRY.FILES | ENTRY.CLIENT_FILES | ENTRY.REFERENCE_FILES
    )


def test_full_reference_acquisition_verifies_every_added_file(tmp_path, monkeypatch):
    raw = b"invented reference software fixture, not benchmark evidence"
    identity = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    monkeypatch.setattr(ENTRY, "FILES", {"scorer.py": identity})
    monkeypatch.setattr(ENTRY, "CLIENT_FILES", {"client.py": identity})
    monkeypatch.setattr(ENTRY, "REFERENCE_FILES", {"timing.py": identity})
    output = tmp_path / "source"
    checked = ENTRY.acquire(
        output, opener=lambda *a, **k: io.BytesIO(raw), include_reference=True
    )
    assert checked["selected_original_files_verified"] == 3
    assert checked["complete_reference_sources_included"]
    assert checked["source_checkbox_eligible"] is False
    (output / "timing.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Git blob mismatch"):
        ENTRY.verify(output, include_reference=True)


def test_distribution_byte_limit_is_not_applied_to_other_files(monkeypatch):
    name = ENTRY.PREFIX + "icc_gt_distribution.json"
    raw = b"x" * ENTRY.FILE_BYTE_LIMITS[name]
    identity = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    monkeypatch.setattr(
        ENTRY, "REFERENCE_FILES", {name: identity, "other.py": identity}
    )
    ENTRY.check_bytes(name, raw)
    with pytest.raises(ValueError, match="Git blob mismatch"):
        ENTRY.check_bytes("other.py", raw)
    with pytest.raises(ValueError, match="Git blob mismatch"):
        ENTRY.check_bytes(name, raw + b"x")


def test_optional_client_verification_requires_unchanged_complete_sources(
    tmp_path, monkeypatch
):
    raw = b"invented software fixture, not a task or audio"
    identity = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    monkeypatch.setattr(ENTRY, "FILES", {"scorer.py": identity})
    monkeypatch.setattr(ENTRY, "CLIENT_FILES", {"client.py": identity})
    root = tmp_path / "source"
    root.mkdir()
    (root / "scorer.py").write_bytes(raw)
    assert ENTRY.verify(root)["selected_original_files_verified"] == 1
    with pytest.raises(FileNotFoundError):
        ENTRY.verify(root, include_client=True)
    (root / "client.py").write_bytes(raw)
    checked = ENTRY.verify(root, include_client=True)
    assert checked["selected_original_files_verified"] == 2
    assert checked["client_sources_included"]
    assert not checked["source_checkbox_eligible"]
    (root / "client.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Git blob mismatch"):
        ENTRY.verify(root, include_client=True)


def test_optional_acquisition_uses_the_same_verifier(tmp_path, monkeypatch):
    raw = b"invented client software fixture"
    identity = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    monkeypatch.setattr(ENTRY, "FILES", {"scorer.py": identity})
    monkeypatch.setattr(ENTRY, "CLIENT_FILES", {"client.py": identity})
    result = ENTRY.acquire(
        tmp_path / "external",
        opener=lambda *a, **k: io.BytesIO(raw),
        include_client=True,
    )
    assert result["selected_original_files_verified"] == 2
    assert result["client_sources_included"]


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
