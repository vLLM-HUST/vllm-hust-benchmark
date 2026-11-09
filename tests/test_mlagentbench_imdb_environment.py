import importlib.util
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "imdb_environment",
    Path(__file__).resolve().parents[1]
    / "scripts/check_mlagentbench_imdb_environment.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_dependencies_are_appended_without_shadowing_base(tmp_path, monkeypatch):
    base = str(tmp_path / "base")
    monkeypatch.setattr(sys, "path", [base])
    MODULE.append_paths([tmp_path, tmp_path])
    assert sys.path == [base, str(tmp_path.resolve())]


def test_dependency_path_requires_directory(tmp_path):
    file = tmp_path / "file"
    file.write_text("not a package directory")
    with pytest.raises(ValueError, match="directory"):
        MODULE.append_paths([file])
