"""Invented saved-answer controls; not original model results or scientific scores."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    "stream_score",
    Path(__file__).resolve().parents[1] / "scripts/score_streamingbench_original.py",
)
runner = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(spec.origin).parent))
try:
    spec.loader.exec_module(runner)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize("task", ["real", "omni", "sqa", "proactive"])
def test_full_original_fields_and_group_shape_preserved(task):
    question = {"question": "query", "answer": "A", "time_stamp": "00:00:01"}
    group = {"video_path": "original.mp4", "questions": [question]}
    original = [[group]] if task == "sqa" else [group]
    saved = deepcopy(original)
    output = saved[0][0]["questions"][0] if task == "sqa" else saved[0]["questions"][0]
    output["Model"] = (
        {"dialog_history": [{"time": 1, "content": "no"}]}
        if task == "proactive"
        else "A. full original output"
    )
    snapshot = deepcopy(saved)
    result = runner.validate_answers(original, saved, model="Model", task=task)
    assert result["original_question_occurrences"] == 1
    assert saved == snapshot
    output["answer"] = "CHANGED"
    with pytest.raises(ValueError, match="changed original"):
        runner.validate_answers(original, saved, model="Model", task=task)


@pytest.mark.parametrize("result", [None, "", [], ["A"]])
def test_empty_or_non_native_result_not_silently_omitted(result):
    original = [{"questions": [{"answer": "A"}]}]
    saved = deepcopy(original)
    saved[0]["questions"][0]["Model"] = result
    with pytest.raises(ValueError, match="empty/malformed"):
        runner.validate_answers(original, saved, model="Model", task="real")


def test_missing_result_and_unsupported_streaming_shape_fail_explicitly():
    original = [{"questions": [{"answer": "A"}]}]
    with pytest.raises(ValueError, match="missing saved"):
        runner.validate_answers(original, original, model="Model", task="real")
    with pytest.raises(ValueError, match="official offline"):
        runner.validate_answers(
            original, original, model="../Model", task="real_stream"
        )
