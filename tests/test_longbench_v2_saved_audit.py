"""Synthetic offline contracts, not original data or native runtime evidence."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


def load():
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location(
            "lb_saved_audit", scripts / "verify_longbench_v2_original_saved.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def fixture():
    original = {"_id": "fixture", "context": "synthetic", "answer": "A"}
    prompt = "synthetic prompt"
    request = {
        "model": "fixture",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 128,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    response = {
        "choices": [{"message": {"content": " B "}, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 150000, "completion_tokens": 128},
    }
    prediction = dict(original, response="B", pred="B", judge=False)
    summary = {
        "id": "fixture",
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "full_prompt_tokens": 150000,
        "completion_tokens": 128,
        "finish_reason": "length",
        "full_input_verified": True,
    }
    return [
        original,
        prompt,
        request,
        response,
        prediction,
        summary,
        lambda text: text,
        "fixture",
    ]


def test_saved_wrong_length_result_is_retained_not_repaired():
    assert load().audit_case(*fixture()) is False


@pytest.mark.parametrize(
    "change", ["prompt", "budget", "thinking", "score", "usage", "reference"]
)
def test_saved_original_profile_or_scoring_tamper_rejected(change):
    args = copy.deepcopy(fixture())
    if change == "prompt":
        args[2]["messages"][0]["content"] = "shortened"
    elif change == "budget":
        args[2]["max_tokens"] = 64
    elif change == "thinking":
        args[2]["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
    elif change == "score":
        args[4]["judge"] = True
    elif change == "usage":
        args[3]["usage"]["prompt_tokens"] = 10
    else:
        args[4]["answer"] = "B"
    with pytest.raises(ValueError):
        load().audit_case(*args)
