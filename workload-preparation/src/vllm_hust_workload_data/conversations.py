"""Explicit original dialogue views; no model, sampling or research analysis."""

from __future__ import annotations

from copy import deepcopy

from .records import WorkloadRecord

HISTORICAL_DIALOGUE_FIELDS = {
    "WILDCHAT": "conversation",
    "ULTRACHAT": "messages",
    "LMSYS_CHAT": "conversation",
    "TULU3": "messages",
    "OPENTHOUGHTS3": "conversations",
}


def _historical_messages(record):
    source_id = record.source.get("source_id")
    messages = record.task["input"].get(HISTORICAL_DIALOGUE_FIELDS[source_id])
    if source_id != "OPENTHOUGHTS3":
        return messages
    if not isinstance(messages, list) or not messages:
        raise ValueError("nonempty original OpenThoughts conversation required")
    roles = {"human": "user", "gpt": "assistant", "system": "system"}
    result = []
    for turn in messages:
        if (
            not isinstance(turn, dict)
            or set(turn) != {"from", "value"}
            or turn["from"] not in roles
            or not isinstance(turn["value"], str)
        ):
            raise ValueError("explicit original OpenThoughts from/value turns required")
        result.append({"role": roles[turn["from"]], "content": turn["value"]})
    return result


def historical_dialogue_views(record: WorkloadRecord) -> list[dict]:
    """Select all original assistant positions and an unanswered final user.

    Validate the entire conversation before returning any view: malformed later
    turns must not silently disappear after an earlier request succeeds.
    """
    source_id = record.source.get("source_id")
    if source_id not in HISTORICAL_DIALOGUE_FIELDS:
        raise ValueError("explicit supported historical dialogue record required")
    messages = _historical_messages(record)
    if not isinstance(messages, list) or not messages:
        raise ValueError("nonempty original conversation required")
    # Validate every turn, including annotations on the final original reply.
    previous = None
    positions = []
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or ("name" in message and not isinstance(message["name"], str))
            or set(message) & {"tool_calls", "function_call", "tool_call_id"}
        ):
            raise ValueError("unsupported original text dialogue turn")
        role = message["role"]
        if role == "system":
            if index != 0:
                raise ValueError("noninitial system turn needs an explicit protocol")
        elif role == "user":
            if previous not in {None, "system", "assistant"}:
                raise ValueError("nonalternating original dialogue")
        elif previous != "user":
            raise ValueError("original assistant must follow user")
        else:
            positions.append(index)
        previous = role
    if previous == "user":
        positions.append(len(messages))
    if not positions:
        raise ValueError("original conversation has no continuation requests")
    return [
        historical_dialogue_prefix(record, index, allow_unanswered_tail=True)
        for index in positions
    ]


def historical_dialogue_prefix(
    record: WorkloadRecord,
    target_message_index: int,
    *,
    allow_unanswered_tail: bool = False,
) -> dict:
    """One original assistant turn's full teacher-forced history, not a live episode.

    These corpora are not original API request captures. Only role/content/name
    are transported; original annotations stay separate. No added system prompt,
    target answer, future turn, tool emulation, schema or budget inference.
    An unanswered final user turn requires explicit opt-in and has no reference.
    """
    source_id = record.source.get("source_id")
    if (
        source_id not in HISTORICAL_DIALOGUE_FIELDS
        or record.runtime.get("kind") != "historical_dialogue"
    ):
        raise ValueError("explicit supported historical dialogue record required")
    messages = _historical_messages(record)
    if (
        not isinstance(messages, list)
        or type(target_message_index) is not int
        or not 1 <= target_message_index <= len(messages)
        or (target_message_index == len(messages) and not allow_unanswered_tail)
    ):
        raise ValueError("explicit original assistant turn position required")
    prefix = messages[:target_message_index]
    target = (
        messages[target_message_index] if target_message_index < len(messages) else None
    )
    if target is not None and (
        not isinstance(target, dict)
        or target.get("role") != "assistant"
        or not isinstance(target.get("content"), str)
    ):
        raise ValueError("original target must be an assistant text turn")
    if any(
        not isinstance(m, dict)
        or m.get("role") not in {"system", "user", "assistant"}
        or not isinstance(m.get("content"), str)
        for m in prefix
    ):
        raise ValueError(
            "original text dialogue contract required; no coercion or flattening"
        )
    if prefix[-1]["role"] != "user":
        raise ValueError(
            "original assistant target must follow a user turn in this profile"
        )
    if any(
        set(m) & {"tool_calls", "function_call", "tool_call_id"}
        for m in [*prefix, *([target] if target is not None else [])]
    ):
        raise ValueError("tool-bearing histories require an explicit tool protocol")
    if any("name" in m and not isinstance(m["name"], str) for m in prefix):
        raise ValueError("invalid original message name")
    api_keys = {"role", "content", "name"}
    native_messages = (
        record.task["input"]["conversations"]
        if source_id == "OPENTHOUGHTS3"
        else messages
    )
    return deepcopy(
        {
            "source_id": source_id,
            "view": "original-historical-assistant-prefix/v1"
            if target is not None
            else "original-final-user-prefix/v1",
            "source": record.source,
            "conversation_id": record.task["task_id"],
            "source_row_index": record.replay["source_row_index"],
            "target_message_index": target_message_index
            if target is not None
            else None,
            "request_after_original_prefix_length": target_message_index,
            "prefix_message_indices": list(range(target_message_index)),
            "messages": [{k: v for k, v in m.items() if k in api_keys} for m in prefix],
            "non_api_prefix_metadata": [
                {k: v for k, v in m.items() if k not in api_keys} for m in prefix
            ],
            "reference": (
                {
                    "kind": "historical_reply_not_quality_gold",
                    "native_message": native_messages[target_message_index],
                }
                if target is not None
                else None
            ),
            "teacher_forced_original_history": True,
            "live_episode": False,
            "original_api_request_capture": False,
            **(
                {
                    "original_prefix_messages": native_messages[:target_message_index],
                    "transport_role_mapping": {
                        "human": "user",
                        "gpt": "assistant",
                        "system": "system",
                    },
                }
                if source_id == "OPENTHOUGHTS3"
                else {}
            ),
        }
    )


def validate_history_targets(source_id, targets, mapping=None):
    if targets is None:
        return
    if (
        source_id != "OASST1"
        or mapping is not None
        or not isinstance(targets, list)
        or not targets
        or any(not isinstance(t, str) or not t for t in targets)
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("history_targets requires distinct OASST original target IDs")


def history_record(rows, target_id, *, source, index):
    view = oasst_history_prefix(rows, target_id)
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": target_id,
            "session_id": view["message_tree_id"],
            "input": {"messages": view["messages"]},
        },
        runtime={"kind": "historical_continuation", "execution": "not_executed"},
        replay={
            "source_row_index": index,
            "mapping": view["view"],
            "prefix_message_ids": view["prefix_message_ids"],
            "synthetic_fallback": False,
        },
        outcome=view["reference"],
    )


def oasst_history_prefix(rows: list[dict], target_message_id: str) -> dict:
    """Restore one complete original path, excluding the target assistant reply.

    This is teacher-forced historical continuation, not an original API request
    or a live conversation episode. Missing links and malformed paths fail closed.
    The caller selects the target; this function does not silently filter rows.
    """
    lookup = {}
    for row in rows:
        key = row.get("message_id")
        if not isinstance(key, str) or not key or key in lookup:
            raise ValueError("OASST requires unique nonempty message IDs")
        lookup[key] = row
    if target_message_id not in lookup:
        raise ValueError("OASST target not found")
    target = lookup[target_message_id]
    if target.get("role") != "assistant":
        raise ValueError("OASST continuation target must be assistant")
    tree = target.get("message_tree_id")
    if not isinstance(tree, str) or not tree:
        raise ValueError("OASST requires an original tree ID")
    chain, seen = [], set()
    current = target
    while True:
        key = current["message_id"]
        if key in seen:
            raise ValueError("OASST cyclic parent links")
        seen.add(key)
        if (
            current.get("message_tree_id") != tree
            or current.get("deleted") is not False
            or current.get("synthetic") is not False
            or not isinstance(current.get("text"), str)
            or not current["text"].strip()
            or "parent_id" not in current
        ):
            raise ValueError("OASST invalid or unavailable original path")
        chain.append(current)
        parent = current["parent_id"]
        if parent is None:
            break
        if not isinstance(parent, str) or parent not in lookup:
            raise ValueError("OASST missing original parent")
        current = lookup[parent]
    chain.reverse()
    if chain[0]["message_id"] != tree or len(chain) < 2 or len(chain) % 2:
        raise ValueError("OASST incomplete original root-to-assistant path")
    for index, row in enumerate(chain):
        if row.get("role") != ("prompter" if index % 2 == 0 else "assistant"):
            raise ValueError("OASST original path must alternate roles")
    return deepcopy(
        {
            "source_id": "OASST1",
            "view": "original-parent-history/v1",
            "target_message_id": target_message_id,
            "message_tree_id": tree,
            "prefix_message_ids": [row["message_id"] for row in chain[:-1]],
            "messages": [
                {
                    "role": "user" if row["role"] == "prompter" else "assistant",
                    "content": row["text"],
                }
                for row in chain[:-1]
            ],
            "reference": {
                "kind": "historical_reply_not_quality_gold",
                "text": target["text"],
            },
        }
    )
