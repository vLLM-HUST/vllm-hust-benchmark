"""Model-free transport contracts, not original dataset/model results."""

import copy
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "chartqa_runner", ROOT / "scripts/run_chartqa_original.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def fixture():
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "original fixture question"}],
        }
    ]
    body = {
        "model": "local-model",
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result = {
        "choices": [{"finish_reason": "stop", "message": {"content": "wrong answer"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }
    arguments = dict(
        model="local-model", messages=copy.deepcopy(messages), disable_thinking=True
    )
    return body, result, arguments


def test_original_selection_and_full_suite():
    assert runner.select_positions(2500) == [0, 1, 2]
    assert runner.select_positions(2500, [1250, 1251, 1252]) == [1250, 1251, 1252]
    assert len(runner.select_positions(2500, full=True)) == 2500


@pytest.mark.parametrize(
    "indices", [[0, 1], [0, 0, 2], [-1, 1, 2], [0, 1, 2500], [True, 1, 2], [2, 1, 0]]
)
def test_no_missing_duplicate_or_invalid_original_positions(indices):
    with pytest.raises(ValueError):
        runner.select_positions(2500, indices)


def test_full_suite_cannot_hide_selected_cases():
    with pytest.raises(ValueError):
        runner.select_positions(2500, [0, 1, 2], full=True)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.openai.com/v1/chat/completions",
        "http://secret@localhost/v1/chat/completions",
        "http://localhost/v1/chat/completions?key=x",
        "http://localhost/v1/completions",
    ],
)
def test_no_cloud_or_credentialled_route(endpoint):
    with pytest.raises(ValueError):
        runner.validate_endpoint(endpoint)


def test_local_route_and_incorrect_answer_preserved():
    runner.validate_endpoint("http://127.0.0.1:19466/v1/chat/completions")
    body, result, arguments = fixture()
    assert runner.validate_completion(body, result, **arguments) == "wrong answer"
    result["choices"][0]["finish_reason"] = "length"
    result["choices"][0]["message"]["content"] = ""
    assert runner.validate_completion(body, result, **arguments) == ""


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 4096),
        ("model", "different"),
        ("messages", []),
        ("temperature", 0.6),
        ("chat_template_kwargs", {"enable_thinking": True}),
    ],
)
def test_changed_original_defaults_or_wire_input_rejected(field, value):
    body, result, arguments = fixture()
    body[field] = value
    with pytest.raises(ValueError):
        runner.validate_completion(body, result, **arguments)


@pytest.mark.parametrize(
    "field,value",
    [
        ("completion_tokens", 2049),
        ("prompt_tokens", 0),
        ("total_tokens", 99),
        ("completion_tokens", True),
    ],
)
def test_bad_usage_cannot_be_completion_evidence(field, value):
    body, result, arguments = fixture()
    result["usage"][field] = value
    with pytest.raises(ValueError):
        runner.validate_completion(body, result, **arguments)


def test_undeclared_thinking_option_not_injected_by_default():
    body, result, arguments = fixture()
    body.pop("chat_template_kwargs")
    arguments["disable_thinking"] = False
    assert runner.validate_completion(body, result, **arguments) == "wrong answer"
