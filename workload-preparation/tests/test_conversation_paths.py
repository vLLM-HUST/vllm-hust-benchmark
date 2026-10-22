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


def test_history_pipeline_roundtrip_and_default_unchanged(tmp_path):
    import hashlib
    import json

    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.consumer import resolve_input
    from vllm_hust_workload_data.pipeline import prepared_rows, run

    root = tmp_path / "source"
    root.mkdir()
    data = root / "messages.jsonl"
    data.write_text(
        "\n".join(json.dumps(row) for row in fixture_rows()), encoding="utf-8"
    )
    manifest = {
        "source_id": "OASST1",
        "revision": "test-fixture",
        "unit": "message",
        "provenance_kind": "test_fixture",
        "files": [
            {
                "path": data.name,
                "role": "data",
                "rows": 4,
                "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            }
        ],
    }
    binding = tmp_path / "binding.json"
    binding.write_text(json.dumps(manifest), encoding="utf-8")
    ordinary = list(iter_workload("OASST1", data_root=root, manifest=manifest))
    assert len(ordinary) == 4
    assert all(r.runtime["kind"] == "historical_message" for r in ordinary)
    config = {
        "sources": [
            {
                "source_id": "OASST1",
                "data_root": str(root),
                "manifest": str(binding),
                "history_targets": ["target"],
            }
        ]
    }
    out = tmp_path / "prepared"
    report = run(config, output=out)
    assert report["sources"][0]["scope"] == "BOUNDED_SLICE"
    database = out / "workload.sqlite"
    refs = list(prepared_rows(database))
    assert len(refs) == 1
    record = resolve_input(database, refs[0]["record_id"], data_root=root)
    assert record.runtime["kind"] == "historical_continuation"
    assert len(record.task["input"]["messages"]) == 3
    assert "excluded reference" not in json.dumps(record.input_record())
    assert record.outcome_record()["text"] == "excluded reference"
    assert "excluded reference".encode() not in database.read_bytes()
    ordinary_config = {
        "sources": [
            {k: v for k, v in config["sources"][0].items() if k != "history_targets"}
        ]
    }
    run(ordinary_config, output=tmp_path / "ordinary")
    ordinary_refs = list(prepared_rows(tmp_path / "ordinary/workload.sqlite"))
    assert refs[0]["record_id"] != ordinary_refs[3]["record_id"]


@pytest.mark.parametrize(
    "source,targets", [("GSM8K", ["a"]), ("OASST1", []), ("OASST1", ["a", "a"])]
)
def test_history_selection_contract(source, targets):
    from vllm_hust_workload_data.conversations import validate_history_targets

    with pytest.raises(ValueError):
        validate_history_targets(source, targets)
