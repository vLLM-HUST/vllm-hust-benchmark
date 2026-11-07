"""Original complete Video-MME-v2 question groups; no prompts or media decoding."""

from copy import deepcopy
from itertools import islice
from pathlib import Path
import re

from .assets import iter_rows
from .records import WorkloadRecord


FIELDS = {
    "video_id",
    "url",
    "group_type",
    "group_structure",
    "question_id",
    "question",
    "options",
    "answer",
    "level",
    "second_head",
    "third_head",
}


def validate_group(group):
    if not isinstance(group, list) or len(group) != 4:
        raise ValueError(
            "four complete original questions required; do not split or replace"
        )
    for row in group:
        if (
            not isinstance(row, dict)
            or set(row) != FIELDS
            or any(not isinstance(row[key], str) for key in FIELDS)
        ):
            raise ValueError(
                "all eleven original string fields required; options and group structure stay opaque strings"
            )
        if not re.fullmatch(r"[0-9]{3}", row["video_id"]) or row["group_type"] not in {
            "logic",
            "relevance",
        }:
            raise ValueError("original video identifier and group type required")
        if any(
            not row[key] for key in ("question", "options", "answer", "group_structure")
        ):
            raise ValueError("original question/options/reference/structure required")
    first = group[0]
    if any(
        any(
            row[key] != first[key]
            for key in ("video_id", "group_type", "group_structure")
        )
        for row in group
    ):
        raise ValueError("original four-question grouping relation differs")
    if [row["question_id"] for row in group] != [
        first["video_id"] + "-" + str(i) for i in range(1, 5)
    ]:
        raise ValueError("original question order or group membership differs")


def group_rows(rows):
    rows = iter(rows)
    seen = set()
    while group := list(islice(rows, 4)):
        validate_group(group)
        identity = group[0]["video_id"]
        if identity in seen:
            raise ValueError(
                "duplicate original group identity; do not deduplicate silently"
            )
        seen.add(identity)
        yield {"original_rows": group}


def original_rows(path):
    path = Path(path)
    if path.name != "test.parquet":
        raise ValueError("original Video-MME-v2 test Parquet required")
    rows = iter_rows(path)
    try:
        yield from group_rows(rows)
    finally:
        rows.close()


def adapt_case(row, *, source, index):
    if source.get("source_id") != "VIDEOMME_V2" or set(row) != {"original_rows"}:
        raise ValueError("original Video-MME-v2 group envelope required")
    group = row["original_rows"]
    validate_group(group)
    identity = group[0]["video_id"]
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": identity,
            "identity_kind": "original_complete_group_video_id",
            "native_unit": "original_four_question_video_group",
            "input": {
                "original_questions": [
                    {
                        key: deepcopy(value)
                        for key, value in original.items()
                        if key != "answer"
                    }
                    for original in group
                ],
                "native_media_refs": {
                    "video": f"videos/{identity}.mp4",
                    "subtitle_archive": "subtitle.zip",
                    "subtitle_member_basename": f"{identity}.jsonl",
                },
            },
        },
        runtime={
            "kind": "native_video_task",
            "execution": "not_executed",
            "media_verified_by_this_adapter": False,
            "requirements": [
                "Keep all four original questions and group structure; options remain the original string, not a parsed or rewritten list",
                "Original video/subtitle binding and native64-frame preprocessing require separate verification",
                "Use original grouped nonlinear rating and accuracy, not independent-row accuracy alone",
                "This envelope is not a rendered model request or proof of native execution",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "videomme_v2_original_complete_groups",
            "source_row_index": index,
            "original_annotation_row_indices": list(range(index * 4, index * 4 + 4)),
            "order": "original_complete_group_order_not_request_arrival",
            "synthetic_fallback": False,
        },
        outcome={
            "references": {
                "original_answers": [
                    {
                        "question_id": original["question_id"],
                        "answer": original["answer"],
                    }
                    for original in group
                ]
            }
        },
    )


def restore_original_group(record):
    if record.source.get("source_id") != "VIDEOMME_V2":
        raise ValueError("original Video-MME-v2 record required")
    inputs = deepcopy(record.task["input"]["original_questions"])
    references = record.outcome["references"]["original_answers"]
    if len(inputs) != 4 or len(references) != 4:
        raise ValueError("all four original inputs and references required")
    for question, reference in zip(inputs, references, strict=True):
        if (
            set(reference) != {"question_id", "answer"}
            or question.get("question_id") != reference["question_id"]
            or "answer" in question
        ):
            raise ValueError("original question/reference binding differs")
        question["answer"] = reference["answer"]
    validate_group(inputs)
    return inputs
