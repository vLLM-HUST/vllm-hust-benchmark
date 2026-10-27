"""Offline acquisition safety tests, not original audio/native model validation."""

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "fleurs_preparer", SCRIPTS / "prepare_fleurs_original_assets.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_original_test_plan_contains_every_real_locale():
    files = module.select_files(module.source_spec("FLEURS"), "test", None)
    assert len(files) == 102
    assert sum(item["rows"] for item in files) == 77810


def test_explicit_locale_selection_does_not_rewrite_files():
    source = module.source_spec("FLEURS")
    selected = module.select_files(source, "test", ["af_za", "am_et"])
    assert len(selected) == 2
    assert all(item in source["files"] for item in selected)
    with pytest.raises(ValueError, match="unknown locale"):
        module.select_files(source, "test", ["all"])


def test_partial_is_never_overwritten(tmp_path):
    target = tmp_path / "original.parquet"
    partial = tmp_path / "original.parquet.partial"
    partial.write_bytes(b"existing unfinished download")
    with pytest.raises(ValueError, match="ownership"):
        module.acquire(target, {}, "https://huggingface.co")
    assert partial.read_bytes() == b"existing unfinished download"
    assert not target.exists()
