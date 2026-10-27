"""Synthetic adapter contracts, not real-data or model validation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


def module():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "narrativeqa_runner", scripts / "run_narrativeqa_original_summary.py"
        )
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.pop(0)


def test_original_summary_and_question_not_ids_or_references_go_to_model():
    task = {
        "input": {
            "summary": " complete synthetic summary ",
            "question": " original question ",
            "document_id": "private-id",
        },
        "outcome": {
            "answer1": "hidden-reference-one",
            "answer2": "hidden-reference-two",
        },
    }
    message = module().messages(task)[0]["content"]
    assert (
        " complete synthetic summary " in message and " original question " in message
    )
    assert "private-id" not in message and "hidden-reference" not in message


def test_metric_keeps_both_references_and_declared_normalization():
    calls = []
    metric = SimpleNamespace(
        get_metric_score=lambda candidate, refs: (
            calls.append((candidate, refs)) or [1, 2, 3, 4, 5, 6]
        )
    )
    runner = module()
    result = runner.metric_scores(
        metric, " Answer. ", {"answer1": "Answer.", "answer2": "Other."}
    )
    assert calls == [("answer", ["answer", "other"])]
    assert result["bleu4"] == 2 and result["meteor"] == 3
    assert runner.metric_scores(
        metric, "", {"answer1": "A", "answer2": "B"}
    ) == dict.fromkeys(runner.METRICS, 0.0)


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_wrong_or_empty_answer_is_not_replaced(finish):
    response = {
        "choices": [{"finish_reason": finish, "message": {"content": ""}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 256, "total_tokens": 356},
    }
    assert module().decode(response, 256) == ""
    response["usage"]["total_tokens"] = 1
    with pytest.raises(ValueError, match="usage"):
        module().decode(response, 256)
