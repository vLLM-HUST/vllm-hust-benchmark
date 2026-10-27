"""Software fixtures, not publisher task executions or delivery evidence."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "workload-preparation/src")
)

from vllm_hust_workload_data.conversations import (
    historical_dialogue_prefix,
    historical_dialogue_views,
)
from vllm_hust_workload_data.records import WorkloadRecord


def record(source="WILDCHAT", messages=None):
    field = "messages" if source in {"ULTRACHAT", "TULU3"} else "conversation"
    return WorkloadRecord(
        source={"source_id": source},
        task={"task_id": "fixture", "input": {field: messages}},
        runtime={"kind": "historical_dialogue"},
        replay={"source_row_index": 7},
    )


@pytest.mark.parametrize("source", ["WILDCHAT", "ULTRACHAT", "LMSYS_CHAT", "TULU3"])
def test_original_prefix_keeps_history_but_excludes_target_and_future(source):
    messages = [
        {"role": "system", "content": " original system "},
        {"role": "user", "content": "first", "language": "en"},
        {"role": "assistant", "content": "historical reply"},
        {"role": "user", "content": " second\n"},
        {"role": "assistant", "content": "held-out answer"},
        {"role": "user", "content": "future"},
    ]
    original = deepcopy(messages)
    item = record(source, messages)
    view = historical_dialogue_prefix(item, 4)
    assert [m["content"] for m in view["messages"]] == [
        m["content"] for m in messages[:4]
    ]
    assert view["non_api_prefix_metadata"][1] == {"language": "en"}
    assert view["reference"]["native_message"] == messages[4]
    assert not view["live_episode"]
    assert not view["original_api_request_capture"]
    view["messages"][0]["content"] = "changed"
    view["reference"]["native_message"]["content"] = "changed"
    assert messages == original
    with pytest.raises(ValueError, match="not a ready"):
        item.request_body()


def test_unanswered_tail_is_explicit_and_has_no_invented_reference():
    item = record("ULTRACHAT", [{"role": "user", "content": "question"}])
    with pytest.raises(ValueError):
        historical_dialogue_prefix(item, 1)
    view = historical_dialogue_prefix(item, 1, allow_unanswered_tail=True)
    assert view["reference"] is None
    assert view["target_message_index"] is None
    assert view["request_after_original_prefix_length"] == 1


def test_openthoughts_original_sharegpt_roles_are_only_transported_not_rewritten():
    original = [
        {"from": "human", "value": " full original question\n"},
        {"from": "gpt", "value": "teacher annotation, not evaluation gold"},
    ]
    snapshot = deepcopy(original)
    item = WorkloadRecord(
        source={"source_id": "OPENTHOUGHTS3"},
        task={"task_id": "fixture", "input": {"conversations": original}},
        runtime={"kind": "historical_dialogue"},
        replay={"source_row_index": 0},
    )
    (view,) = historical_dialogue_views(item)
    assert view["messages"] == [{"role": "user", "content": original[0]["value"]}]
    assert view["original_prefix_messages"] == original[:1]
    assert view["reference"]["native_message"] == original[1]
    assert view["reference"]["kind"] == "historical_reply_not_quality_gold"
    assert "teacher annotation" not in str(view["messages"])
    assert original == snapshot
    original.append({"from": "tool", "value": "unsupported later turn"})
    with pytest.raises(ValueError, match="from/value"):
        historical_dialogue_views(item)


def test_all_turns_keep_original_positions_and_teacher_forced_history():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "original first answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "original second answer"},
        {"role": "user", "content": "unanswered"},
    ]
    views = historical_dialogue_views(record(messages=messages))
    assert [v["request_after_original_prefix_length"] for v in views] == [1, 3, 5]
    assert views[-1]["reference"] is None
    assert views[1]["messages"][1] == messages[1]
    assert views[0]["messages"] == messages[:1]


@pytest.mark.parametrize(
    "tail",
    [
        {"role": "assistant", "content": "duplicate assistant"},
        {"role": "tool", "content": "requires real tool protocol"},
        {"role": "system", "content": "late system"},
        {"role": "user", "content": None},
    ],
)
def test_malformed_later_turn_prevents_partial_selection(tail):
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        tail,
    ]
    with pytest.raises(ValueError):
        historical_dialogue_views(record(messages=messages))


@pytest.mark.parametrize("index", [True, -1, 0, 3, "1"])
def test_invalid_original_position_rejected(index):
    item = record(
        messages=[
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    )
    with pytest.raises(ValueError):
        historical_dialogue_prefix(item, index)


@pytest.mark.parametrize(
    "mutation",
    [
        {"content": [{"type": "text", "text": "no flattening"}]},
        {"tool_calls": []},
        {"function_call": None},
        {"tool_call_id": "call"},
        {"name": 1},
        {"role": "tool"},
    ],
)
def test_unsupported_original_message_fails_without_coercion(mutation):
    user = {"role": "user", "content": "question", **mutation}
    item = record(messages=[user, {"role": "assistant", "content": "answer"}])
    with pytest.raises(ValueError):
        historical_dialogue_prefix(item, 1)
