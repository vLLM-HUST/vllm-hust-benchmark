"""Inert input/prediction protocol fixtures, never real task evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "dureader_original", ROOT / "scripts/run_dureader_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def row():
    return {
        "question_id": 42,
        "question": "original\u2028question",
        "question_type": "ENTITY",
        "documents": [
            {
                "title": "original title",
                "paragraphs": ["first original paragraph", "second original paragraph"],
                "is_selected": True,
                "most_related_para": 1,
            }
        ],
        "answers": ["secret gold"],
        "entity_answers": [["secret entity"]],
        "answer_spans": [[1, 2]],
    }


def test_all_original_text_is_input_but_no_gold_annotations():
    original = row()
    frozen = deepcopy(original)
    messages = runner.messages(original)
    body = json.loads(messages[1]["content"])
    assert body == {
        "question": original["question"],
        "documents": [
            {
                "title": "original title",
                "paragraphs": ["first original paragraph", "second original paragraph"],
            }
        ],
    }
    assert "secret" not in json.dumps(messages)
    assert "is_selected" not in messages[1]["content"]
    assert original == frozen


def test_native_references_keep_empty_and_original_aliases():
    original = row()
    assert runner.reference(original)["answers"] is original["answers"]
    assert runner.reference(original)["entity_answers"] is original["entity_answers"]
    assert runner.reference(original)["yesno_answers"] == []
    original["answers"] = []
    assert runner.reference(original)["answers"] == []


def test_predictions_are_not_repaired_or_replaced_with_gold():
    original = row()
    result = runner.prediction(
        '{"answers":["wrong answer"],"yesno_answers":[],"entity_answers":[[]]}',
        original,
    )
    assert result["answers"] == ["wrong answer"]
    assert result["question_type"] == "ENTITY"
    for invalid in [
        "not JSON",
        "```json\n{}\n```",
        "{}",
        '{"answers":[],"yesno_answers":[],"entity_answers":[[]]}',
        '{"answers":["x"],"yesno_answers":["maybe"],"entity_answers":[[]]}',
    ]:
        with pytest.raises(ValueError):
            runner.prediction(invalid, original)


def test_fixed_selection_no_failed_case_swaps():
    rows = [row() for _ in range(100)]
    assert [i for i, _ in runner.select(rows)] == [0, 1, 2]
    assert len(runner.select(rows, full=True)) == 100
    assert [i for i, _ in runner.select(rows, start=97)] == [97, 98, 99]
    for kwargs in (
        {"start": True},
        {"count": 2},
        {"start": 98},
        {"full": True, "count": 100},
    ):
        with pytest.raises(ValueError):
            runner.select(rows, **kwargs)
