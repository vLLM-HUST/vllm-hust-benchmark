"""Original MVBench video-view records and media references, no inference."""

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from .records import WorkloadRecord


def relative_media(prefix, video):
    for value in (prefix, video):
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError("Original relative POSIX media locator required")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or ":" in value:
            raise ValueError("Media locator cannot escape external asset root")
    return str(PurePosixPath(prefix) / video)


def original_rows(path):
    path = Path(path)
    if path.name != "test.json":
        raise ValueError("Original video-view test.json required")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"root", "meta"}:
        raise ValueError("Unknown original root fields require explicit mapping")
    if (
        not isinstance(raw["root"], dict)
        or not isinstance(raw["meta"], dict)
        or set(raw["root"]) != set(raw["meta"])
    ):
        raise ValueError("Original per-task media roots and records must agree")
    global_index = 0
    for task, cases in raw["meta"].items():
        if not isinstance(task, str) or not task or not isinstance(cases, list):
            raise ValueError("Original ordered task groups required")
        for offset, original in enumerate(cases):
            base_fields = {"video", "question", "answer", "candidates"}
            schemas = [
                base_fields,
                base_fields | {"start", "end"},
                base_fields | {"start", "end", "accurate_start", "accurate_end"},
                base_fields | {"start", "end", "fps", "show_name", "subtitle"},
            ]
            if not isinstance(original, dict) or set(original) not in schemas:
                raise ValueError(
                    "Unknown original case fields require explicit mapping"
                )
            choices = original["candidates"]
            for name in {
                "start",
                "end",
                "accurate_start",
                "accurate_end",
            } & original.keys():
                if type(original[name]) not in (int, float):
                    raise ValueError("Original temporal annotation type required")
            if "fps" in original and type(original["fps"]) is not int:
                raise ValueError("Original fps annotation type required")
            if any(
                not isinstance(original[name], str)
                for name in {"show_name", "subtitle"} & original.keys()
            ):
                raise ValueError("Original subtitle/show annotation type required")
            if (
                not isinstance(original["question"], str)
                or not original["question"].strip()
                or not isinstance(choices, list)
                or not choices
                or any(not isinstance(item, str) or not item for item in choices)
                or not isinstance(original["answer"], str)
                or original["answer"] not in choices
            ):
                raise ValueError(
                    "Original question/options/reference types and relation required"
                )
            yield {
                "original": deepcopy(original),
                "task_type": task,
                "original_task_offset": offset,
                "original_global_index": global_index,
                "media_root": raw["root"][task],
                "media_path": relative_media(raw["root"][task], original["video"]),
            }
            global_index += 1


def adapt_case(row, *, source, index):
    original = deepcopy(row["original"])
    if row["media_path"] != relative_media(row["media_root"], original["video"]):
        raise ValueError("Original media binding differs")
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": f"{row['task_type']}:{row['original_task_offset']}",
            "identity_kind": "original_file_group_and_position_not_publisher_task_id",
            "native_unit": "video_multiple_choice_task",
            "input": {
                "question": original["question"],
                "candidates": original["candidates"],
                "video": original["video"],
                "media_path": row["media_path"],
            },
        },
        runtime={
            "kind": "native_video_task",
            "execution": "not_executed",
            "requirements": [
                "Original media, frame sampling, prompt builder and scorer required; not a ready text request"
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "mvbench_original_video_tasks",
            "source_row_index": index,
            "order": "original_group_and_case_order_not_arrival_time",
            "annotations": {
                key: deepcopy(row[key])
                for key in (
                    "task_type",
                    "original_task_offset",
                    "original_global_index",
                    "media_root",
                    "media_path",
                )
            },
            "synthetic_fallback": False,
            "original_case_annotations": {
                key: deepcopy(value)
                for key, value in original.items()
                if key not in {"video", "question", "answer", "candidates"}
            },
            "unmapped_field_names": [],
        },
        outcome={
            "answer": original["answer"],
            "reference_kind": "original_option_text_not_observed_prediction",
        },
    )
