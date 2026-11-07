"""Original Video-MME annotation projection; no decoding, model or analysis."""

from copy import deepcopy
from pathlib import Path
import re

from .assets import iter_rows
from .records import WorkloadRecord

FIELDS = {
    "video_id",
    "duration",
    "domain",
    "sub_category",
    "url",
    "videoID",
    "question_id",
    "task_type",
    "question",
    "options",
    "answer",
}
INPUT_FIELDS = {"question", "options", "videoID"}


def validate_case(row):
    if not isinstance(row, dict) or set(row) != FIELDS:
        raise ValueError("Unknown original Video-MME fields require explicit mapping")
    if any(not isinstance(row[key], str) for key in FIELDS - {"options"}):
        raise ValueError("Original Video-MME string field types required")
    if any(not row[key].strip() for key in ("question", "video_id", "question_id")):
        raise ValueError("Original question and identifiers required")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", row["videoID"]):
        raise ValueError("Contained original video filename identifier required")
    if row["duration"] not in {"short", "medium", "long"}:
        raise ValueError("Original duration label required")
    if (
        not isinstance(row["options"], list)
        or len(row["options"]) != 4
        or any(not isinstance(choice, str) or not choice for choice in row["options"])
    ):
        raise ValueError("All four original options in original order required")
    if row["answer"] not in {"A", "B", "C", "D"}:
        raise ValueError("Original reference option letter required")


def original_rows(path):
    if Path(path).name != "test-00000-of-00001.parquet":
        raise ValueError("Original Video-MME test Parquet required")
    for row in iter_rows(Path(path)):
        validate_case(row)
        yield row


def adapt_case(row, *, source, index):
    validate_case(row)
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": row["question_id"],
            "identity_kind": "original_question_id",
            "native_unit": "video_multiple_choice_task",
            "input": {
                **{key: deepcopy(row[key]) for key in INPUT_FIELDS},
                "native_media_refs": {
                    "video": f"video/{row['videoID']}.mp4",
                    "subtitle": f"subtitle/{row['videoID']}.srt",
                },
            },
        },
        runtime={
            "kind": "native_video_task",
            "execution": "not_executed",
            "requirements": [
                "Original media, native frame/subtitle preprocessing and scorer required separately; not a ready text or image request",
                "A missing original subtitle follows the selected native profile's original behavior, not fabricated text",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "videomme_original_tasks",
            "source_row_index": index,
            "order": "original_annotation_order_not_request_arrival",
            "original_annotations": {
                key: deepcopy(row[key]) for key in FIELDS - INPUT_FIELDS - {"answer"}
            },
            "media_verified_by_this_adapter": False,
            "unmapped_field_names": [],
            "synthetic_fallback": False,
        },
        outcome={
            "answer": row["answer"],
            "reference_kind": "original_option_letter_not_observed_prediction",
        },
    )
