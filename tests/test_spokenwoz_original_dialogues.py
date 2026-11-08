"""Invented structural fixtures; real original CPU audit is separate evidence."""

from copy import deepcopy
import json

import pytest

from vllm_hust_workload_data.spokenwoz import (
    adapt_case,
    original_rows,
    restore_original_case,
)
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def fixture():
    turn = {
        "tag": "user",
        "text": "fixture request",
        "words": [
            {
                "Word": "fixture",
                "BeginTime": 0,
                "EndTime": 1,
                "ChannelId": 0,
                "Word_index": 0,
            }
        ],
        "metadata": {},
        "dialog_act": {},
        "span_info": [],
    }
    system = {
        **deepcopy(turn),
        "tag": "system",
        "text": "fixture response",
        "metadata": {"fixture": "state reference"},
    }
    return {
        "dialogue_id": "fixture",
        "original_dialogue": {
            "goal": {"fixture": "goal reference"},
            "log": [turn, system],
        },
    }


def test_complete_dialogue_roundtrip_and_reference_separation():
    row = fixture()
    before = deepcopy(row)
    record = adapt_case(row, source={"source_id": "SPOKENWOZ"}, index=0)
    identity, restored = restore_original_case(record)
    assert identity == row["dialogue_id"] and restored == row["original_dialogue"]
    assert row == before
    assert "goal reference" not in json.dumps(record.input_record())
    assert "state reference" not in json.dumps(record.input_record())
    assert (
        record.task["input"]["turns"][0]["words"]
        == row["original_dialogue"]["log"][0]["words"]
    )
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_file_order_and_complete_dialogues_preserved(tmp_path):
    path = tmp_path / "fixture.json"
    data = {k: fixture()["original_dialogue"] for k in ["second", "first"]}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert [r["dialogue_id"] for r in original_rows(path)] == ["second", "first"]


@pytest.mark.parametrize("mode", ["odd", "new_label", "bad_role"])
def test_partial_or_unreviewed_contract_rejected(mode):
    row = fixture()
    if mode == "odd":
        row["original_dialogue"]["log"].pop()
    elif mode == "new_label":
        row["original_dialogue"]["log"][0]["future_answer"] = "do not leak"
    else:
        row["original_dialogue"]["log"][1]["tag"] = "user"
    with pytest.raises(ValueError):
        adapt_case(row, source={"source_id": "SPOKENWOZ"}, index=0)


def test_catalog_binds_original_public_release_not_generated_chat():
    source = source_spec("SPOKENWOZ")
    assert source["repo_id"] == "ssz1111/SpokenWOZ-Test-Text-Fixed"
    assert source["files"][0]["bytes"] == 52233254
    assert adapter_spec("SPOKENWOZ")["mapping"] == "spokenwoz_original_dialogues"
