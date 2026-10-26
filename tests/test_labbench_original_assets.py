"""Small artificial ZIP fixtures only; no original tasks or network calls."""

import importlib.util
from pathlib import Path
import sys
from zipfile import ZipFile, ZipInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "labbench_assets", ROOT / "scripts/prepare_labbench_original_assets.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_preserves_original_bytes_and_refuses_existing_source(tmp_path):
    archive = tmp_path / "fixture.zip"
    payload = b"original\r\n"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("publisher/data/fixture.txt", payload)
    source = tmp_path / "source"
    assert (
        MODULE.extract_original(
            archive, source, expected_files=1, expected_bytes=len(payload)
        )
        == 1
    )
    assert (source / "data/fixture.txt").read_bytes() == payload
    with pytest.raises(ValueError, match="fresh"):
        MODULE.extract_original(
            archive, source, expected_files=1, expected_bytes=len(payload)
        )


@pytest.mark.parametrize("name", ["publisher/../escape", "publisher/a\\escape"])
def test_rejects_unsafe_archive_before_writing(tmp_path, name):
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zipped:
        info = ZipInfo("fixture")
        info.filename = name
        zipped.writestr(info, b"x")
    source = tmp_path / "source"
    with pytest.raises(ValueError, match="unsafe"):
        MODULE.extract_original(archive, source, expected_files=1, expected_bytes=1)
    assert not source.exists()
