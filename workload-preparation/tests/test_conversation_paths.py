"""Small invented fixtures test contracts, never substitute for real source evidence."""

from copy import deepcopy

import pytest

from vllm_hust_workload_data.conversations import oasst_history_prefix


def fixture_rows():
    return [
        dict(
            message_id=key,
            parent_id=parent,
            role=role,
            text=text,
            message_tree_id="root",
            deleted=False,
            synthetic=False,
        )
        for key, parent, role, text in [
            ("root", None, "prompter", "original\n question "),
            ("a", "root", "assistant", "earlier reply"),
            ("u", "a", "prompter", "follow-up"),
            ("target", "u", "assistant", "excluded reference"),
        ]
    ]


def test_parent_order_not_file_order_and_target_is_separate():
    rows = fixture_rows()
    before = deepcopy(rows)
    result = oasst_history_prefix(list(reversed(rows)), "target")
    assert rows == before
    assert result["prefix_message_ids"] == ["root", "a", "u"]
    assert [m["role"] for m in result["messages"]] == ["user", "assistant", "user"]
    assert result["messages"][0]["content"] == "original\n question "
    assert result["reference"]["text"] == "excluded reference"
    assert all(m["content"] != "excluded reference" for m in result["messages"])
    result["messages"][0]["content"] = "mutated"
    assert rows == before


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "missing",
        "cycle",
        "tree",
        "role",
        "deleted",
        "synthetic",
        "text",
        "root",
    ],
)
def test_invalid_paths_fail_without_payload_in_error(failure):
    rows = fixture_rows()
    if failure == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif failure == "missing":
        rows.pop(1)
    elif failure == "cycle":
        rows[0]["parent_id"] = "u"
    elif failure == "tree":
        rows[1]["message_tree_id"] = "other"
    elif failure == "role":
        rows[1]["role"] = "prompter"
    elif failure in {"deleted", "synthetic"}:
        rows[1][failure] = True
    elif failure == "text":
        rows[1]["text"] = " "
    else:
        rows[0]["message_tree_id"] = "other"
    with pytest.raises(ValueError) as error:
        oasst_history_prefix(rows, "target")
    assert "excluded reference" not in str(error.value)


def test_unknown_or_nonassistant_target_rejected():
    for target in ["missing", "u"]:
        with pytest.raises(ValueError):
            oasst_history_prefix(fixture_rows(), target)
