"""Inert ToolACE-shape fixtures and mocked transport; no real model/tool calls."""

import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_toolace_original_continuation")


def original():
    return {
        "system": "Original tool descriptions.",
        "conversations": [
            {"from": "user", "value": "Original question"},
            {"from": "assistant", "value": "Original textual function call"},
            {"from": "tool", "value": "Original historical tool response"},
            {"from": "assistant", "value": "Original second reply"},
            {"from": "user", "value": "Original followup"},
            {"from": "assistant", "value": "Original third reply"},
        ],
    }


def test_every_original_assistant_position_keeps_tool_history_without_ids():
    row = original()
    views = runner.original_prefixes(row)
    assert [view["original_assistant_turn_index"] for view in views] == [1, 3, 5]
    assert views[0]["messages"] == [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": "Original question"},
    ]
    assert views[1]["messages"][-1] == {
        "role": "tool",
        "content": "Original historical tool response",
    }
    assert views[1]["original_tool_turn_indices"] == [2]
    assert views[2]["historical_reference"] == "Original third reply"
    assert "Original third reply" not in json.dumps(views[2]["messages"])
    assert all(
        set(turn) == {"role", "content"} for view in views for turn in view["messages"]
    )
    views[0]["messages"][0]["content"] = "changed view"
    assert row["system"] == "Original tool descriptions."


@pytest.mark.parametrize(
    "failure",
    [
        "role",
        "extra-field",
        "content-type",
        "no-system",
        "first-role",
        "consecutive-assistant",
    ],
)
def test_unsupported_original_shape_fails_before_model_call(failure):
    row = original()
    if failure == "role":
        row["conversations"][-1]["from"] = "invented"
    elif failure == "extra-field":
        row["conversations"][-1]["tool_call_id"] = "invented"
    elif failure == "content-type":
        row["conversations"][-1]["value"] = ["parts"]
    elif failure == "no-system":
        row.pop("system")
    elif failure == "first-role":
        row["conversations"][0]["from"] = "tool"
    else:
        row["conversations"][2]["from"] = "assistant"
    with pytest.raises(ValueError):
        runner.original_prefixes(row)


def test_fixed_range_full_and_no_small_or_invalid_corpus():
    rows = [original()] * 11300
    assert [i for i, _ in runner.select(rows)] == [0, 1, 2]
    assert [i for i, _ in runner.select(rows, start=3)] == [3, 4, 5]
    assert len(runner.select(rows, full=True)) == 11300
    for count in (0, 1, 2):
        with pytest.raises(ValueError):
            runner.select(rows, count=count)
    with pytest.raises(ValueError):
        runner.select(rows[:3])
    with pytest.raises(ValueError):
        runner.select(rows, full=True, start=1)


def result(text="new text", finish="stop", tokens=3):
    return {
        "choices": [{"text": text, "finish_reason": finish}],
        "usage": {"prompt_tokens": 10, "completion_tokens": tokens},
    }


def invoke(tmp_path, transport, **kwargs):
    return runner.run(
        [(0, original())],
        render=lambda messages: (json.dumps(messages), 10),
        output=tmp_path / "private",
        endpoint="http://127.0.0.1:19466/v1",
        model="fixture",
        max_tokens=32,
        context=128,
        transport=transport,
        **kwargs,
    )


def test_original_raw_rendered_requests_complete_without_tool_execution(tmp_path):
    calls = []

    def transport(url, body):
        calls.append((url, body))
        return {"count": 10} if url.endswith("/tokenize") else result()

    receipt = invoke(tmp_path, transport)
    assert receipt["conversations"] == 1 and len(receipt["requests"]) == 3
    assert (
        receipt["tools_executed"]
        == receipt["dynamic_observations"]
        == receipt["invented_tool_call_ids"]
        == 0
    )
    assert not receipt["quality_scored"] and not receipt["source_checkbox_eligible"]
    requests = [body for url, body in calls if url.endswith("/completions")]
    assert len(requests) == 3 and "tools" not in requests[0]
    assert json.loads(requests[1]["prompt"])[-1]["role"] == "tool"


@pytest.mark.parametrize(
    "text,finish,tokens", [("", "stop", 0), ("unfinished", "length", 32)]
)
def test_empty_or_budget_outputs_retained_without_quality_claim(
    tmp_path, text, finish, tokens
):
    def transport(url, body):
        return (
            {"count": 10} if url.endswith("/tokenize") else result(text, finish, tokens)
        )

    receipt = invoke(tmp_path, transport)
    assert [r["finish_reason"] for r in receipt["requests"]] == [finish] * 3
    saved = json.loads(
        (tmp_path / "private/request-000000.private.json").read_text("utf-8")
    )
    assert saved["response"]["choices"][0]["text"] == text
    assert not receipt["quality_scored"]


@pytest.mark.parametrize(
    "failure", ["tokenizer", "capacity", "response", "transport", "final-identity"]
)
def test_failure_preserves_original_attempt_and_never_retries_or_replaces(
    tmp_path, failure
):
    calls = []

    def transport(url, body):
        calls.append(url)
        if failure == "transport":
            raise RuntimeError("inert failure")
        if url.endswith("/tokenize"):
            return {
                "count": 11
                if failure == "tokenizer"
                else 120
                if failure == "capacity"
                else 10
            }
        response = result()
        if failure == "response":
            response["choices"] = []
        return response

    def verify():
        if failure == "final-identity":
            raise ValueError("source changed")

    with pytest.raises((RuntimeError, ValueError)):
        invoke(tmp_path, transport, final_verify=verify)
    receipt = json.loads((tmp_path / "private/receipt-private.json").read_text("utf-8"))
    assert receipt["status"] == "FAILED_NOT_COMPLETE"
    assert (tmp_path / "private/original-row-0.private.json").exists()
    assert not receipt["source_checkbox_eligible"]
    assert len(calls) <= 6


def test_remote_endpoint_refused_without_creating_results(tmp_path):
    with pytest.raises(ValueError):
        runner.run(
            [(0, original())],
            render=lambda _: ("fixture", 10),
            output=tmp_path / "never",
            endpoint="https://external.example/v1",
            model="fixture",
            max_tokens=32,
            context=128,
        )
    assert not (tmp_path / "never").exists()
