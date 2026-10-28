"""Software transport fixtures only, not original dataset/model/scorer runs."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "ib_runner", ROOT / "scripts/run_infinitebench_original.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tmp_path, *, tokens=10, fail=False):
    calls = []

    def messages(tokenizer, row, task, data_dir, model_name):
        assert tokenizer is None
        assert model_name == "author-template"
        assert data_dir == str(tmp_path)
        return [{"role": "user", "content": row["context"]}], row["context"]

    def complete(**body):
        calls.append(body)
        if fail:
            raise RuntimeError("fixture transport failure")
        return NS(
            choices=[
                NS(finish_reason="stop", message=NS(content="fixture prediction"))
            ],
            model_dump=lambda **_: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "fixture prediction"},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    utils = NS(
        MODEL_TO_PROMPT_TEMPLATE={"author-template": {}},
        DATA_NAME_TO_MAX_NEW_TOKENS={"fixture": 50},
        create_msgs=messages,
        get_answer=lambda row, _: row["answer"],
    )
    scorer = NS(get_score=lambda labels, predictions, task, model: 0.0)
    kwargs = dict(
        task="fixture",
        utils=utils,
        scorer=scorer,
        client=NS(chat=NS(completions=NS(create=complete))),
        token_count=lambda _: tokens,
        model="local-model",
        template="author-template",
        context_limit=100,
        output=tmp_path / "output",
        data_dir=tmp_path,
    )
    rows = [
        {"context": f"original fixture {i}", "answer": f"private reference {i}"}
        for i in range(5)
    ]
    return rows, kwargs, calls


def test_original_positions_budget_prompt_and_scorer_preserved(tmp_path):
    rows, kwargs, calls = fixture(tmp_path)
    result = runner.execute(rows, **kwargs)
    assert result["selected_original_positions"] == [0, 1, 2]
    assert result["native_execution_complete"]
    assert not result["source_checkbox_eligible"]
    assert result["score"] == 0.0
    assert all(call["max_tokens"] == 50 for call in calls)
    assert [call["messages"][0]["content"] for call in calls] == [
        row["context"] for row in rows[:3]
    ]
    assert all("private reference" not in json.dumps(call) for call in calls)
    assert result["attempted_original_positions"] == [0, 1, 2]
    assert len(result["private_protocol_records"]) == 3
    for item in result["private_protocol_records"]:
        index = item["row_index"]
        request = kwargs["output"] / f"row-{index}.request-private.json"
        response = kwargs["output"] / f"row-{index}.response-private.json"
        assert (
            hashlib.sha256(request.read_bytes()).hexdigest() == item["request_sha256"]
        )
        assert (
            hashlib.sha256(response.read_bytes()).hexdigest() == item["response_sha256"]
        )
        assert json.loads(request.read_text()) == calls[index]
        assert json.loads(response.read_text())["usage"]["completion_tokens"] == 5


def test_oversized_context_is_not_trimmed_or_replaced(tmp_path):
    rows, kwargs, calls = fixture(tmp_path, tokens=51)
    with pytest.raises(ValueError, match="no trimming"):
        runner.execute(rows, **kwargs)
    assert calls == []
    receipt = json.loads((kwargs["output"] / "receipt-private.json").read_text())
    assert receipt["status"] == "FAILED_NOT_COMPLETE"
    assert not receipt["native_execution_complete"]


def test_explicit_nonthinking_transport_keeps_original_task(tmp_path):
    rows, kwargs, calls = fixture(tmp_path)
    result = runner.execute(rows, **kwargs, disable_thinking=True)
    assert result["model_chat_template_kwargs"] == {"enable_thinking": False}
    assert all(
        call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
        for call in calls
    )
    assert all(call["max_tokens"] == 50 for call in calls)
    assert [call["messages"][0]["content"] for call in calls] == [
        row["context"] for row in rows[:3]
    ]


def test_transport_failure_does_not_retry_or_switch_question(tmp_path):
    rows, kwargs, calls = fixture(tmp_path, fail=True)
    with pytest.raises(RuntimeError):
        runner.execute(rows, **kwargs)
    assert len(calls) == 1


def test_full_and_minimum_selection_rules(tmp_path):
    rows, _, _ = fixture(tmp_path)
    assert runner.select(rows, full=True) == list(enumerate(rows))
    with pytest.raises(ValueError):
        runner.select(rows, count=2)
    with pytest.raises(ValueError):
        runner.select(rows, full=True, start=1)


def test_scorer_failure_preserves_predictions_without_completion(tmp_path):
    rows, kwargs, calls = fixture(tmp_path)

    def failed_score(*_):
        raise RuntimeError("fixture scoring failure")

    kwargs["scorer"] = NS(get_score=failed_score)
    with pytest.raises(RuntimeError):
        runner.execute(rows, **kwargs)
    assert len(calls) == 3
    receipt = json.loads((kwargs["output"] / "receipt-private.json").read_text())
    assert receipt["completed_original_positions"] == [0, 1, 2]
    assert receipt["status"] == "FAILED_NOT_COMPLETE"
    assert not receipt["native_execution_complete"]
    assert (
        len((kwargs["output"] / "predictions-private.jsonl").read_text().splitlines())
        == 3
    )


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1, True, "0.5"])
def test_invalid_aggregate_never_counts_as_complete(tmp_path, score):
    rows, kwargs, _ = fixture(tmp_path)
    kwargs["scorer"] = NS(get_score=lambda *_: score)
    with pytest.raises(ValueError, match="invalid aggregate"):
        runner.execute(rows, **kwargs)
    receipt = json.loads((kwargs["output"] / "receipt-private.json").read_text())
    assert not receipt["native_execution_complete"]


def test_failure_on_second_row_preserves_first_and_never_skips(tmp_path):
    rows, kwargs, calls = fixture(tmp_path)
    original = kwargs["client"].chat.completions.create

    def second_fails(**body):
        if calls:
            raise RuntimeError("fixture second row failure")
        return original(**body)

    kwargs["client"].chat.completions.create = second_fails
    with pytest.raises(RuntimeError):
        runner.execute(rows, **kwargs)
    receipt = json.loads((kwargs["output"] / "receipt-private.json").read_text())
    assert receipt["selected_original_positions"] == [0, 1, 2]
    assert receipt["completed_original_positions"] == [0]
    assert receipt["attempted_original_positions"] == [0, 1]
    assert receipt["private_protocol_records"][1]["response_sha256"] is None
    assert (kwargs["output"] / "row-1.request-private.json").is_file()
    assert not receipt["native_execution_complete"]


@pytest.mark.parametrize("reason,content", [("tool_calls", "text"), ("stop", None)])
def test_invalid_model_response_is_not_success(tmp_path, reason, content):
    rows, kwargs, _ = fixture(tmp_path)
    kwargs["client"].chat.completions.create = lambda **_: NS(
        choices=[NS(finish_reason=reason, message=NS(content=content))],
        model_dump=lambda **_: {
            "choices": [{"finish_reason": reason, "message": {"content": content}}]
        },
    )
    with pytest.raises(ValueError, match="unsuccessful"):
        runner.execute(rows, **kwargs)
