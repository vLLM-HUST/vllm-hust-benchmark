"""Software field contracts, never fabricated benchmark evidence."""

from copy import deepcopy

import pytest

from vllm_hust_workload_data.videomme import adapt_case, validate_case
from vllm_hust_workload_data.registry import adapter_spec, source_spec


def row():
    return {
        "video_id": "001",
        "duration": "short",
        "domain": "fixture-domain",
        "sub_category": "fixture-category",
        "url": "https://example.invalid/fixture",
        "videoID": "fixture-id",
        "question_id": "001-1",
        "task_type": "fixture-task",
        "question": "Fixture question only",
        "options": ["A. first", "B. second", "C. third", "D. fourth"],
        "answer": "B",
    }


def test_all_original_fields_reconstruct_and_reference_is_not_input():
    original = row()
    record = adapt_case(original, source={"source_id": "VIDEOMME"}, index=7)
    restored = {
        key: record.task["input"][key] for key in ("question", "options", "videoID")
    }
    restored.update(record.replay["original_annotations"])
    restored["answer"] = record.outcome["answer"]
    assert restored == original
    assert "answer" not in record.task["input"]
    assert record.task["task_id"] == "001-1"
    assert record.replay["source_row_index"] == 7
    assert record.task["input"]["native_media_refs"]["video"] == "video/fixture-id.mp4"
    assert (
        record.task["input"]["native_media_refs"]["subtitle"]
        == "subtitle/fixture-id.srt"
    )
    assert record.runtime["execution"] == "not_executed"
    assert not record.replay["media_verified_by_this_adapter"]
    record.task["input"]["options"].append("changed")
    assert original == row()


@pytest.mark.parametrize(
    "update",
    [
        {"videoID": "../escape"},
        {"videoID": "/absolute"},
        {"videoID": "a\\b"},
        {"video_id": 1},
        {"answer": "E"},
        {"options": ["one"]},
        {"extra": "unmapped"},
        {"question": ""},
    ],
)
def test_unknown_fields_bad_types_and_media_escape_are_rejected(update):
    original = deepcopy(row())
    original.update(update)
    with pytest.raises(ValueError):
        validate_case(original)


def test_manifest_registers_all_original_archives_but_only_one_task_table():
    source = source_spec("VIDEOMME")
    assert len(source["files"]) == 24
    assert sum(f["bytes"] for f in source["files"]) == 101002235118
    assert [f["path"] for f in source["files"] if f["role"] == "data"] == [
        "videomme/test-00000-of-00001.parquet"
    ]
    assert sum(f["role"] == "source_archive" for f in source["files"]) == 21
    assert adapter_spec("VIDEOMME")["mapping"] == "videomme_original_tasks"
    assert source["access"]["license"] == "custom-academic-research-only"
