import json

import pytest

from vllm_hust_workload_data.mvbench import adapt_case, original_rows, relative_media


def fixture(path, **updates):
    item = {
        "video": "clip.mp4",
        "question": "Which event?",
        "candidates": ["first", "second"],
        "answer": "second",
    }
    item.update(updates)
    path.write_text(
        json.dumps({"root": {"action": "videos"}, "meta": {"action": [item, item]}}),
        encoding="utf-8",
    )


def test_original_order_duplicates_media_and_reference_are_preserved(tmp_path):
    path = tmp_path / "test.json"
    fixture(path)
    rows = list(original_rows(path))
    assert [r["original_global_index"] for r in rows] == [0, 1]
    assert [r["original_task_offset"] for r in rows] == [0, 1]
    assert rows[0]["media_path"] == "videos/clip.mp4"
    record = adapt_case(rows[0], source={"source_id": "MVBENCH"}, index=0)
    assert "answer" not in record.task["input"]
    assert record.outcome["answer"] == "second"
    assert record.task["input"]["candidates"] == ["first", "second"]
    record.task["input"]["candidates"].append("changed")
    assert len(rows[0]["original"]["candidates"]) == 2


@pytest.mark.parametrize(
    "value", ["../escape.mp4", "/absolute.mp4", "C:/drive.mp4", "a\\b.mp4"]
)
def test_media_paths_cannot_escape_original_root(value):
    with pytest.raises(ValueError):
        relative_media("videos", value)


def test_original_temporal_and_subtitle_annotations_are_not_lost(tmp_path):
    path = tmp_path / "test.json"
    extras = {
        "start": 0.0,
        "end": 15.0,
        "fps": 3,
        "show_name": "original-show",
        "subtitle": "original subtitle",
    }
    fixture(path, **extras)
    row = next(original_rows(path))
    record = adapt_case(row, source={"source_id": "MVBENCH"}, index=0)
    assert record.replay["original_case_annotations"] == extras
    assert "subtitle" not in record.task["input"]


@pytest.mark.parametrize(
    "updates",
    [
        {"answer": "missing"},
        {"candidates": "not-list"},
        {"extra": "unreviewed"},
        {"question": ""},
    ],
)
def test_invalid_or_unreviewed_fields_fail_instead_of_synthetic_fallback(
    tmp_path, updates
):
    path = tmp_path / "test.json"
    fixture(path, **updates)
    with pytest.raises(ValueError):
        list(original_rows(path))
