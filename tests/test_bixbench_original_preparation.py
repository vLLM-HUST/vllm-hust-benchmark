"""Preparation boundary fixtures, not scientific benchmark evidence."""

import hashlib
import importlib.util
from pathlib import Path
import zipfile

import pytest

ENTRY = (
    Path(__file__).resolve().parents[1] / "scripts/prepare_bixbench_original_tasks.py"
)


def module():
    spec = importlib.util.spec_from_file_location("bixbench_prepare", ENTRY)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_plan_needs_no_data_author_or_runtime(tmp_path):
    report = module().prepare(
        *(tmp_path / name for name in ("author", "data", "output", "workspace"))
    )
    assert report["status"] == "PLAN_ONLY" and report["model_calls"] == 0
    assert not report["source_checkbox_eligible"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("limit", [0, -1, 206])
def test_invalid_prefix_rejected(tmp_path, limit):
    with pytest.raises(ValueError, match="prefix limit"):
        module().prepare(tmp_path, tmp_path, tmp_path, tmp_path, limit)


def test_asset_identity_mismatch_rejected(tmp_path):
    path = tmp_path / "fixture.jsonl"
    path.write_bytes(b"{}\n")
    entry = {
        "path": path.name,
        "bytes": 3,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert module().verify_asset(tmp_path, entry) == path
    path.write_bytes(b"[]\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        module().verify_asset(tmp_path, entry)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "directory\\escape"])
def test_unsafe_zip_members_rejected_without_extraction(tmp_path, name):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("fixture")
        info.filename = name
        archive.writestr(info, "fixture only")
    with pytest.raises(ValueError, match="unsafe original ZIP"):
        module().verify_zip(path)
    assert list(tmp_path.iterdir()) == [path]


def test_safe_zip_checked_without_extraction(tmp_path):
    path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("capsule/Data/fixture.txt", "fixture only")
    module().verify_zip(path)
    assert list(tmp_path.iterdir()) == [path]
