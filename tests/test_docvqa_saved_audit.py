"""Software fixtures only, not publisher samples or scientific score evidence."""

import asyncio
import base64
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "docvqa_saved", scripts / "verify_docvqa_original_saved.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture():
    png = b"fixture-image-not-real-publisher-data"
    image = "data:image/png;base64," + base64.b64encode(png).decode()
    row = {"question": "fixture question", "answers": ["fixture reference"]}
    request = {
        "model": "Qwen/Qwen3.5-9B",
        "max_tokens": 16384,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "fixture template: fixture question"},
                    {"type": "image_url", "image_url": {"url": image}},
                ],
            }
        ],
    }
    response = {
        "choices": [
            {"message": {"content": "fixture result"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
    event = SimpleNamespace(
        event="model",
        input=[
            SimpleNamespace(
                role="user",
                content=[
                    SimpleNamespace(text="fixture template: fixture question"),
                    SimpleNamespace(image=image),
                ],
            )
        ],
        call=SimpleNamespace(request=request, response=response),
        output=SimpleNamespace(completion="fixture result"),
    )
    sample = SimpleNamespace(
        id="49153",
        error=None,
        target=deepcopy(row["answers"]),
        input=[
            SimpleNamespace(
                role="user",
                content=[
                    SimpleNamespace(text=row["question"]),
                    SimpleNamespace(image=image),
                ],
            )
        ],
        events=[event],
        output=SimpleNamespace(completion="fixture result"),
        scores={"docvqa_scorer": SimpleNamespace(value=0.75, answer="fixture parsed")},
    )

    async def scorer(state, target):
        assert state.output.completion == "fixture result"
        assert target == row["answers"]
        return SimpleNamespace(value=0.75, answer="fixture parsed")

    native = SimpleNamespace(
        FREEFORM_TEMPLATE="fixture template: {question}", docvqa_scorer=lambda: scorer
    )
    return sample, row, png, native


def test_saved_request_media_and_unchanged_scorer_reconstruction():
    sample, row, png, native = fixture()
    result = asyncio.run(
        runner.check_sample(sample, row, png, native, lambda value: value)
    )
    assert result == {
        "question_id": "49153",
        "score": 0.75,
        "finish_reason": "stop",
        "input_tokens": 11,
        "output_tokens": 7,
        "native_model_input_image_bytes_verified": True,
        "raw_api_image_log_redacted_not_byte_verifiable": False,
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "reference",
        "image",
        "question",
        "budget",
        "api_prompt",
        "response",
        "score",
        "extra_call",
    ],
)
def test_saved_audit_rejects_changed_input_output_protocol_or_score(mutation):
    sample, row, png, native = fixture()
    event = sample.events[0]
    if mutation == "reference":
        sample.target = ["changed fixture"]
    elif mutation == "image":
        png = b"changed image"
    elif mutation == "question":
        row["question"] = "changed question"
    elif mutation == "budget":
        event.call.request["max_tokens"] = 1
    elif mutation == "api_prompt":
        event.call.request["messages"][0]["content"][0]["text"] = "changed prompt"
    elif mutation == "response":
        event.call.response["choices"][0]["message"]["content"] = "changed output"
    elif mutation == "score":
        sample.scores["docvqa_scorer"].value = 1.0
    else:
        sample.events.append(event)
    with pytest.raises(ValueError):
        asyncio.run(runner.check_sample(sample, row, png, native, lambda value: value))


@pytest.mark.parametrize("changed_model_image", [False, True])
def test_native_api_redaction_does_not_skip_recorded_model_image_validation(
    changed_model_image,
):
    sample, row, png, native = fixture()
    event = sample.events[0]
    event.call.request["messages"][0]["content"][1]["image_url"]["url"] = (
        "<base64-data-removed>"
    )
    if changed_model_image:
        event.input[0].content[1].image = (
            "data:image/png;base64," + base64.b64encode(b"changed").decode()
        )
        with pytest.raises(ValueError, match="recorded native model input"):
            asyncio.run(
                runner.check_sample(sample, row, png, native, lambda value: value)
            )
    else:
        result = asyncio.run(
            runner.check_sample(sample, row, png, native, lambda value: value)
        )
        assert result["native_model_input_image_bytes_verified"] is True
        assert result["raw_api_image_log_redacted_not_byte_verifiable"] is True


def test_saved_audit_contains_no_model_or_network_execution_calls():
    import ast

    tree = ast.parse((scripts / "verify_docvqa_original_saved.py").read_text("utf-8"))
    names = {
        getattr(node.func, "id", getattr(node.func, "attr", ""))
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not names & {"get_model", "generate", "native_eval", "urlopen", "post"}
