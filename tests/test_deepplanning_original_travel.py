"""Travel asset and protocol fixtures; not native execution evidence."""

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "deepplanning_travel", ROOT / "scripts/run_deepplanning_original_travel.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selected_original_travel_databases_only(tmp_path):
    archive = tmp_path / "database.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("database_en/id_0/flights/flights.csv", b"original fixture")
        stream.writestr("database_en/id_1/trains/trains.csv", b"unselected fixture")
    output = tmp_path / "out"
    MODULE.extract_cases(archive, output, "en", [0])
    assert (output / "id_0/flights/flights.csv").read_bytes() == b"original fixture"
    assert not (output / "id_1").exists()
    with pytest.raises(ValueError, match="missing"):
        MODULE.extract_cases(archive, tmp_path / "missing", "en", [5])


def test_traversal_rejected_not_written(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("database_en/id_0/../../escape", b"bad fixture")
    with pytest.raises(ValueError, match="unsafe"):
        MODULE.extract_cases(archive, tmp_path / "out", "en", [0])
    assert not (tmp_path / "escape").exists()


def test_original_conversion_and_scoring_not_omitted():
    source = (ROOT / "scripts/run_deepplanning_original_travel.py").read_text("utf-8")
    assert "max_llm_calls=400" in source
    assert "rerun_ids=ids" in source
    assert "conversion = convert_reports(" in source
    assert "scores = evaluate_plans(" in source
    assert '"qwen-plus": local' in source
    assert "not reproduction" in source
    assert "max_tokens" not in source
    assert "enable_thinking" not in source
