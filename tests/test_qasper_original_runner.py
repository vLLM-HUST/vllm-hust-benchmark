"""Inert original-structure protocol fixtures, no real task/model results."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "qasper_original", ROOT / "scripts/run_qasper_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    question = {
        "question_id": "original-question",
        "question": " original question? ",
        "answers": [
            {
                "answer": {
                    "unanswerable": False,
                    "free_form_answer": "hidden reference",
                    "evidence": ["hidden evidence annotation"],
                }
            }
        ],
        "nlp_background": "original annotation",
    }
    paper = {
        "title": "original title",
        "abstract": "original abstract",
        "full_text": [
            {"section_name": None, "paragraphs": [" original paragraph\nsecond line "]}
        ],
        "figures_and_tables": [{"file": "figure.png", "caption": "original caption"}],
        "qas": [question, {**question, "question_id": "other-original"}],
    }
    return paper, question


def test_complete_paper_and_question_no_reference_or_sibling_leakage():
    paper, question = fixture()
    frozen = deepcopy((paper, question))
    body = json.loads(runner.messages(paper, question)[1]["content"])
    assert set(body) == {
        "title",
        "abstract",
        "full_text",
        "figures_and_tables",
        "question",
    }
    assert body["full_text"] == paper["full_text"]
    assert body["figures_and_tables"] == paper["figures_and_tables"]
    assert body["question"] == question["question"]
    assert "hidden reference" not in json.dumps(body)
    assert "other-original" not in json.dumps(body)
    assert (paper, question) == frozen


def test_selected_gold_keeps_all_original_annotations_and_paper_relations():
    paper, question = fixture()
    papers = {"original-paper": paper}
    selected = [(7, ("original-paper", 0, question))]
    gold = runner.subset_gold(papers, selected)
    assert gold["original-paper"]["qas"] == [question]
    assert gold["original-paper"]["full_text"] == paper["full_text"]
    assert len(paper["qas"]) == 2
    gold["original-paper"]["qas"][0]["answers"].clear()
    assert question["answers"]


def test_strict_native_predictions_not_repaired():
    assert runner.prediction(
        '{"predicted_answer":"wrong answer","predicted_evidence":[]}'
    ) == {"predicted_answer": "wrong answer", "predicted_evidence": []}
    for text in (
        "not JSON",
        "```json\n{}\n```",
        "{}",
        '{"predicted_answer":"x","predicted_evidence":[1]}',
        '{"predicted_answer":[],"predicted_evidence":[]}',
    ):
        with pytest.raises(ValueError):
            runner.prediction(text)


def test_scoring_uses_original_full_evidence_and_counts_missing_predictions():
    gold = {"original-paper": {"qas": []}}
    refs = {
        "original-question": [
            {"answer": "ref", "evidence": ["FLOAT SELECTED: original caption"]}
        ]
    }

    def references(original, text_only):
        assert original is gold and text_only is False
        return refs

    def evaluate(original, predictions):
        assert original is refs and predictions == {}
        return {"Answer F1": 0, "Evidence F1": 0, "Missing predictions": 1}

    module = SimpleNamespace(get_answers_and_evidence=references, evaluate=evaluate)
    assert runner.score(module, gold, [])["Missing predictions"] == 1
    with pytest.raises(ValueError, match="unknown"):
        runner.score(
            module,
            gold,
            [
                {
                    "question_id": "unknown",
                    "predicted_answer": "x",
                    "predicted_evidence": [],
                }
            ],
        )


def test_selection_fixed_before_quality_no_failed_case_swaps():
    rows = list(range(1005))
    assert [i for i, _ in runner.select(rows)] == [0, 1, 2]
    assert [i for i, _ in runner.select(rows, start=1002)] == [1002, 1003, 1004]
    assert len(runner.select(rows, full=True)) == 1005
    for invalid in (
        {"start": True},
        {"count": 2},
        {"start": 1003},
        {"full": True, "count": 4},
    ):
        with pytest.raises(ValueError):
            runner.select(rows, **invalid)
