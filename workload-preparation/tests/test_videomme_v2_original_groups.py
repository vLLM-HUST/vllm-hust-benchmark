"""Opaque-string/group/reference fixtures, not real source tasks or models."""

from copy import deepcopy
import hashlib

import pytest

from vllm_hust_workload_data.videomme_v2 import (
    adapt_case,
    group_rows,
    restore_original_group,
)


def group(identity="001"):
    return [
        {
            "video_id": identity,
            "url": "https://example.invalid/original-fixture",
            "group_type": "logic",
            "group_structure": "[1,[2,3],4]",
            "question_id": f"{identity}-{i}",
            "question": f"fixture question{i}",
            "options": "A. option one\\nB. option two",
            "answer": "A",
            "level": "level_1",
            "second_head": "fixture category",
            "third_head": "fixture subtype",
        }
        for i in range(1, 5)
    ]


def test_all_original_fields_roundtrip_and_answers_do_not_enter_input():
    original = group()
    record = adapt_case(
        {"original_rows": original}, source={"source_id": "VIDEOMME_V2"}, index=0
    )
    assert restore_original_group(record) == original
    assert len(record.task["input"]["original_questions"]) == 4
    assert all(
        "answer" not in row for row in record.task["input"]["original_questions"]
    )
    assert (
        record.task["input"]["original_questions"][0]["options"]
        == original[0]["options"]
    )
    assert record.replay["original_annotation_row_indices"] == [0, 1, 2, 3]
    assert not record.runtime["media_verified_by_this_adapter"]
    with pytest.raises(ValueError, match="not a ready model request"):
        record.request_body()
    record.task["input"]["original_questions"][0]["question"] = "caller change"
    assert original == group()


def test_group_order_is_not_sorted_or_collapsed():
    original = group("009") + group("001")
    packets = list(group_rows(original))
    assert [row["original_rows"][0]["video_id"] for row in packets] == ["009", "001"]
    second = adapt_case(packets[1], source={"source_id": "VIDEOMME_V2"}, index=1)
    assert second.replay["original_annotation_row_indices"] == [4, 5, 6, 7]


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "reorder",
        "cross_video",
        "parse_options",
        "parse_structure",
        "unknown_field",
        "missing_answer",
    ],
)
def test_invalid_or_rewritten_native_group_is_rejected(mutation):
    rows = group()
    if mutation == "partial":
        rows.pop()
    elif mutation == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "cross_video":
        rows[1]["video_id"] = "002"
    elif mutation == "parse_options":
        rows[0]["options"] = ["option one", "option two"]
    elif mutation == "parse_structure":
        rows[0]["group_structure"] = [1, [2, 3], 4]
    elif mutation == "unknown_field":
        rows[0]["extra"] = "unmapped"
    else:
        del rows[0]["answer"]
    with pytest.raises(ValueError):
        list(group_rows(rows))


def test_duplicate_native_groups_are_not_silently_deduplicated():
    with pytest.raises(ValueError, match="duplicate"):
        list(group_rows(group() + group()))


def test_reference_question_binding_is_checked():
    packet = adapt_case(
        {"original_rows": group()}, source={"source_id": "VIDEOMME_V2"}, index=0
    )
    changed = deepcopy(packet)
    changed.outcome["references"]["original_answers"][0]["question_id"] = "002-1"
    with pytest.raises(ValueError, match="binding"):
        restore_original_group(changed)


def test_public_iterator_counts_native_groups_without_relabeling_raw_rows(
    tmp_path, monkeypatch
):
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data import videomme_v2

    path = tmp_path / "test.parquet"
    path.write_bytes(b"opaque parser fixture")
    manifest = {
        "source_id": "VIDEOMME_V2",
        "revision": "fixture",
        "unit": "source_rows",
        "provenance_kind": "fixture",
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "role": "data",
                "rows": 4,
            }
        ],
    }
    monkeypatch.setattr(
        videomme_v2, "original_rows", lambda _path: iter([{"original_rows": group()}])
    )
    records = list(iter_workload("VIDEOMME_V2", data_root=tmp_path, manifest=manifest))
    assert len(records) == 1 and manifest["files"][0]["rows"] == 4
    assert restore_original_group(records[0]) == group()
    manifest["files"][0]["rows"] = 8
    with pytest.raises(ValueError, match="source row count mismatch"):
        list(iter_workload("VIDEOMME_V2", data_root=tmp_path, manifest=manifest))
