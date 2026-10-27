"""Saved-run verification fixtures, not real model or author-scoring results."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "chartqa_saved", ROOT / "scripts/verify_chartqa_original_saved.py"
)
saved = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(saved)


def receipt():
    return {
        "source_id": "CHARTQA",
        "status": "SELECTED_ORIGINAL_VISUAL_TASKS_AND_TOOLKIT_SCORE_COMPLETE_NOT_SOURCE_DELIVERY",
        "selected_original_positions": [0, 1, 2],
        "attempted_original_positions": [0, 1, 2],
        "completed_original_positions": [0, 1, 2],
        "private_records": [{"original_position": index} for index in [0, 1, 2]],
        "official_api_budget": 2048,
        "model_chat_template_kwargs": {"enable_thinking": False},
        "source_proof": {"revision": saved.REVISION, "git_tree": saved.TREE},
    }


def test_saved_original_complete_selection():
    assert saved.validate_receipt(receipt()) == [0, 1, 2]


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "FAILED_RETAINED"),
        ("completed_original_positions", [0, 1]),
        ("attempted_original_positions", [0, 1]),
        ("private_records", [{"original_position": 0}]),
        ("official_api_budget", 4096),
        ("source_proof", {}),
        ("model_chat_template_kwargs", {"enable_thinking": True}),
    ],
)
def test_partial_or_changed_run_never_becomes_complete(field, value):
    original = receipt()
    original[field] = value
    with pytest.raises(ValueError):
        saved.validate_receipt(original)


def case():
    original = {
        "query": "original fixture",
        "label": ["reference"],
        "human_or_machine": 0,
    }
    prediction = {
        "index": 0,
        "question": "original fixture",
        "answer": json.dumps(["reference"]),
        "category": 0,
        "prediction": "wrong",
    }
    protocol = {
        "original_position": 0,
        "prompt_tokens": 10,
        "completion_tokens": 1,
        "finish_reason": "length",
    }
    messages = [{"role": "user", "content": "original fixture"}]
    request = {
        "model": "local-model",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = {
        "choices": [{"message": {"content": "wrong"}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }
    return (
        original,
        prediction,
        protocol,
        request,
        response,
        dict(
            model="local-model", messages=copy.deepcopy(messages), disable_thinking=True
        ),
    )


def test_wrong_answers_and_length_are_preserved_without_model_calls():
    original, prediction, protocol, request, response, arguments = case()
    saved.verify_case(original, prediction, protocol, request, response, **arguments)


@pytest.mark.parametrize(
    "field,value",
    [
        ("question", "replaced"),
        ("answer", '["replaced"]'),
        ("category", 1),
        ("prediction", "replaced"),
        ("index", 1),
    ],
)
def test_prediction_or_original_relationship_changes_rejected(field, value):
    original, prediction, protocol, request, response, arguments = case()
    prediction[field] = value
    with pytest.raises(ValueError):
        saved.verify_case(
            original, prediction, protocol, request, response, **arguments
        )


def test_protocol_usage_changes_rejected():
    original, prediction, protocol, request, response, arguments = case()
    protocol["completion_tokens"] = 99
    with pytest.raises(ValueError, match="usage"):
        saved.verify_case(
            original, prediction, protocol, request, response, **arguments
        )


def test_metrics_use_saved_categories_and_keep_zero_scores():
    saved.compare_metrics([{"Overall": 0.0, "0": 0.0}], [{"Overall": 0.0, 0: 0.0}])
    with pytest.raises(ValueError, match="score differs"):
        saved.compare_metrics([{"Overall": 0.0}], [{"Overall": 100.0}])
    with pytest.raises(ValueError, match="categories"):
        saved.compare_metrics([{"Overall": 0.0, "0": 0.0}], [{"Overall": 0.0}])


def test_missing_overall_metric_cannot_pass():
    with pytest.raises(ValueError, match="overall metric"):
        saved.compare_metrics([{}], [{}])


def test_default_plan_does_not_require_assets_service_or_output(tmp_path):
    output = tmp_path / "must-not-exist"
    arguments = [sys.executable, str(ROOT / "scripts/verify_chartqa_original_saved.py")]
    for flag in (
        "--official-checkout",
        "--source-file",
        "--prepared-view",
        "--ffmpeg-library-dir",
        "--run",
    ):
        arguments += [flag, str(tmp_path / "missing")]
    result = subprocess.run(
        arguments + ["--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    proof = json.loads(result.stdout)
    assert proof["new_model_calls"] == proof["paid_judge_calls"] == 0
    assert not output.exists()
