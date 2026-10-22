"""CPU-only invented protocol fixtures, not QuALITY model evaluation evidence."""

import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/run_quality_original_mc.py"
SPEC = importlib.util.spec_from_file_location("quality_original_mc_cpu", PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_help_without_optional_model_or_scoring_dependencies():
    result = subprocess.run(
        [sys.executable, str(PATH), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--tokenizer" in result.stdout


def test_select_first_question_from_three_distinct_articles_before_execution():
    parents = [
        {"article_id": "fixture-a", "questions": [{}, {}]},
        {"article_id": "fixture-a", "questions": [{}]},
        {"article_id": "fixture-b", "questions": [{}]},
        {"article_id": "fixture-c", "questions": [{}, {}]},
    ]
    chosen, total = runner.select_original_questions(parents)
    assert total == 6
    assert [x[0] for x in chosen] == [0, 3, 4]
    assert [x[1:3] for x in chosen] == [(0, 0), (2, 0), (3, 0)]


def test_gold_and_reference_annotations_never_enter_model_document():
    parent = {"article": "  entire\noriginal article "}
    question = {
        "question": "unaltered question",
        "options": [" one ", "two", "three", "four"],
        "gold_label": 2,
    }

    def normalize(value):
        return " ".join(value.split())

    document = runner.original_document(parent, question, normalize)
    assert document["text"] == parent["article"]
    assert document["question"] == question["question"]
    assert document["choices"] == ["one", "two", "three", "four"]
    assert document["gold"] == 1
    inputs = runner.model_input_document(document)
    assert set(inputs) == {"text", "question", "choices"}
    question["gold_label"] = 4
    assert inputs == runner.model_input_document(
        runner.original_document(parent, question, normalize)
    )


@pytest.mark.parametrize("gold", [True, 0, 5, "2", None])
def test_bad_gold_label_not_coerced(gold):
    with pytest.raises(ValueError):
        runner.original_document(
            {"article": "fixture"},
            {"question": "q", "options": ["a", "b", "c", "d"], "gold_label": gold},
            lambda x: x,
        )


@pytest.mark.parametrize(
    "value",
    [
        "https://outside.example/v1/completions",
        "http://localhost/v1/chat/completions",
        "http://localhost/v1/completions?key=x",
    ],
)
def test_endpoint_rejection(value):
    with pytest.raises(ValueError):
        runner.endpoint(value)


def test_mc_receipt_is_not_free_generation_or_full_dataset_accuracy():
    path = ROOT / "docs/workload-data/validation/quality-original-mc-npu-20261023.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert (
        receipt["public_runner_sha256"]
        == hashlib.sha256(PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    )
    assert receipt["file_sha256"] == runner.DATA_SHA
    assert receipt["harness_task_sha256"] == runner.TASK_SHA
    assert receipt["unique_original_questions"] == len(receipt["samples"]) == 3
    assert (
        receipt["selected_flat_question_indices"]
        == [r["flat_question_index"] for r in receipt["samples"]]
        == [0, 6, 15]
    )
    assert (
        receipt["successful_option_calls_final_run"]
        == sum(r["option_calls"] for r in receipt["samples"])
        == 12
    )
    assert receipt["successful_option_calls_including_same_three_probe_repeat"] == 24
    assert sum(r["acc"] for r in receipt["samples"]) == receipt["raw_correct"] == 1
    assert (
        sum(r["acc_norm"] for r in receipt["samples"])
        == receipt["length_normalized_correct"]
        == 2
    )
    assert receipt["input_truncation"] is False
    assert receipt["chat_template"] is False
    assert receipt["official_publisher_scoring_source"] is False
    assert receipt["full_suite_validated"] is False
    assert receipt["source_checkbox_eligible"] is False
