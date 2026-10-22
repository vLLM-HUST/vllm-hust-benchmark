"""Explicit original OASST tree projection; no sampling, model, or research analysis."""

from __future__ import annotations

from copy import deepcopy


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
