"""Inert contract fixtures only, not original execution or native metric evidence."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "cceval_native_scoring", SCRIPTS / "score_crosscodeeval_original.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_original_fields_and_empty_predictions_preserved():
    original = [
        {"prompt": "inert", "groundtruth": "answer", "metadata": {"task_id": "x"}}
    ]
    prediction = [{**original[0], "task_id": "x", "pred": ""}]
    MODULE.validate_predictions(original, prediction)
    for key, value in [
        ("groundtruth", "changed"),
        ("task_id", "other"),
        ("pred", None),
    ]:
        altered = [{**prediction[0], key: value}]
        with pytest.raises(ValueError, match="fields"):
            MODULE.validate_predictions(original, altered)
    with pytest.raises(ValueError, match="all selected"):
        MODULE.validate_predictions(original, [])


def test_fixed_original_selection_not_replacement(tmp_path):
    archive = tmp_path / "fixture.tar.xz"
    rows = [{"inert": index} for index in range(5)]
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    with tarfile.open(archive, "w:xz") as bundle:
        item = tarfile.TarInfo("python/line_completion.jsonl")
        item.size = len(raw)
        bundle.addfile(item, io.BytesIO(raw))
    assert (
        MODULE.selected_originals(archive, "python", [1, 2, 3], "line_completion")
        == rows[1:4]
    )
    for positions in [[], [0, 2], [-1], [True], [4, 5]]:
        with pytest.raises(ValueError):
            MODULE.selected_originals(archive, "python", positions, "line_completion")
