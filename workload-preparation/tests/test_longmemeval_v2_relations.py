"""Inert memory relation fixtures; not original questions, models or scoring."""

import hashlib
import json

import pytest

from vllm_hust_workload_data.longmemeval_v2 import (
    adapt_case,
    iter_haystack_trajectories,
    original_rows,
    restore_original_question,
)


def fixture(tmp_path):
    questions = [
        {
            "id": "q",
            "domain": "enterprise",
            "environment": "fixture",
            "question_type": "fixture",
            "question": "fixture query",
            "image": None,
            "answer": "private-reference-marker",
            "eval_function": "fixture_reference_only",
        }
    ]
    memories = [
        {
            "id": identity,
            "domain": "enterprise",
            "environment": "fixture",
            "goal": "fixture historical goal",
            "outcome": "failure",
            "start_url": "https://example.invalid",
            "states": [
                {
                    "state_index": 0,
                    "step": 0,
                    "url": "https://example.invalid",
                    "action": None,
                    "thought": "original historical thought",
                    "accessibility_tree": "original fixture tree",
                    "screenshot": f"screenshots/{identity}/0.png",
                }
            ],
        }
        for identity in ["second", "first"]
    ]
    files = []
    values = {
        "questions.jsonl": "\n".join(json.dumps(q) for q in questions) + "\n",
        "trajectories.jsonl": "\n".join(json.dumps(t) for t in memories) + "\n",
        "haystacks/lme_v2_small.json": json.dumps({"q": ["first", "second"]}),
        "haystacks/lme_v2_medium.json": json.dumps({"q": ["second", "first"]}),
        "trajectory_screenshots/enterprise_screenshots_base.tar.gz": "inert archive metadata fixture",
    }
    for name, value in values.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value.encode()
        path.write_bytes(raw)
        files.append(
            {
                "path": name,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "role": "data" if name == "questions.jsonl" else "native_asset",
            }
        )
    return questions, memories, {"relation_files": files}


def test_question_references_both_haystacks_and_native_memory_roundtrip(tmp_path):
    questions, memories, spec = fixture(tmp_path)
    joined = list(original_rows(tmp_path / "questions.jsonl", spec))
    record = adapt_case(joined[0], source={"source_id": "LONGMEMEVAL_V2"}, index=0)
    assert restore_original_question(record) == questions[0]
    assert "private-reference-marker" not in json.dumps(record.input_record())
    assert "fixture_reference_only" not in json.dumps(record.input_record())
    assert record.task["input"]["haystacks"] == {
        "small": ["first", "second"],
        "medium": ["second", "first"],
    }
    assert list(
        iter_haystack_trajectories(record, tier="small", data_root=tmp_path)
    ) == list(reversed(memories))
    assert (
        list(iter_haystack_trajectories(record, tier="medium", data_root=tmp_path))
        == memories
    )
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()


def test_changed_haystack_bytes_rejected(tmp_path):
    _, _, spec = fixture(tmp_path)
    (tmp_path / "haystacks/lme_v2_small.json").write_text('{"q":["second"]}')
    with pytest.raises(ValueError, match="identity mismatch"):
        list(original_rows(tmp_path / "questions.jsonl", spec))


def test_changed_record_order_is_not_original_memory(tmp_path):
    _, _, spec = fixture(tmp_path)
    joined = list(original_rows(tmp_path / "questions.jsonl", spec))
    record = adapt_case(joined[0], source={"source_id": "LONGMEMEVAL_V2"}, index=0)
    record.task["input"]["haystacks"]["small"].reverse()
    with pytest.raises(ValueError, match="order changed"):
        list(iter_haystack_trajectories(record, tier="small", data_root=tmp_path))


@pytest.mark.parametrize("replacement", [["missing"], ["first", "first"]])
def test_invalid_original_relations_rejected_even_with_matching_hash(
    tmp_path, replacement
):
    _, _, spec = fixture(tmp_path)
    path = tmp_path / "haystacks/lme_v2_small.json"
    raw = json.dumps({"q": replacement}).encode()
    path.write_bytes(raw)
    entry = next(
        item
        for item in spec["relation_files"]
        if item["path"] == "haystacks/lme_v2_small.json"
    )
    entry.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="unresolved or cross-domain"):
        list(original_rows(tmp_path / "questions.jsonl", spec))


def test_modified_reference_cannot_restore_original_memory(tmp_path):
    _, _, spec = fixture(tmp_path)
    row = next(original_rows(tmp_path / "questions.jsonl", spec))
    record = adapt_case(row, source={"source_id": "LONGMEMEVAL_V2"}, index=0)
    record.outcome["references"]["answer"] = "changed"
    with pytest.raises(ValueError, match="question no longer matches"):
        list(iter_haystack_trajectories(record, tier="small", data_root=tmp_path))
