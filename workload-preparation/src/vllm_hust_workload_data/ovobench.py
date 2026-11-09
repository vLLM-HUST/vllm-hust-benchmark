"""Preserve native online-video cases and separate stage-level scorer labels."""

from copy import deepcopy

from .records import WorkloadRecord

TASKS = {
    "EPM",
    "HLD",
    "ASI",
    "STU",
    "OJR",
    "ATR",
    "FPD",
    "ACR",
    "OCR",
    "CRR",
    "SSR",
    "REC",
}
FORWARD = {"CRR", "SSR", "REC"}
INPUT_FIELDS = {"task", "video", "question", "options", "realtime", "activity"}
STAGE_INPUT_FIELDS = {"realtime", "step"}
STAGE_REFERENCE_FIELDS = {"type", "count"}


def adapt_case(row, *, source, index):
    if (
        source.get("source_id") != "OVOBENCH"
        or not isinstance(row, dict)
        or type(row.get("id")) is not int
        or row["id"] < 0
        or row.get("task") not in TASKS
        or not isinstance(row.get("video"), str)
        or not row["video"]
    ):
        raise ValueError("original OVO-Bench case identity and video required")
    inputs = {key: deepcopy(value) for key, value in row.items() if key in INPUT_FIELDS}
    references = {
        key: deepcopy(value)
        for key, value in row.items()
        if key not in INPUT_FIELDS | {"id", "test_info"}
    }
    if row["task"] in FORWARD:
        stages = row.get("test_info")
        if not isinstance(stages, list) or not stages:
            raise ValueError("all original forward stages required")
        inputs["test_info"], references["test_info"] = [], []
        for stage in stages:
            if (
                not isinstance(stage, dict)
                or "realtime" not in stage
                or set(stage) - STAGE_INPUT_FIELDS - STAGE_REFERENCE_FIELDS
            ):
                raise ValueError("unrecognized original forward stage fields")
            inputs["test_info"].append(
                {
                    key: deepcopy(value)
                    for key, value in stage.items()
                    if key in STAGE_INPUT_FIELDS
                }
            )
            references["test_info"].append(
                {
                    key: deepcopy(value)
                    for key, value in stage.items()
                    if key in STAGE_REFERENCE_FIELDS
                }
            )
    elif "test_info" in row:
        raise ValueError("unexpected forward stages in backward/realtime case")
    return WorkloadRecord(
        source=deepcopy(source),
        task={"task_id": row["id"], "input": inputs},
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Use original video/chunk sequence and native prompt/scorer",
                "Forward cases require every original test_info stage; labels are reference-only",
                "No static-image/chat fallback or automatic model execution",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "ovobench_original_tasks",
            "source_row_index": index,
            "order": "original_file_order_not_request_arrival",
            "original_field_names": list(row),
            "synthetic_fallback": False,
        },
        outcome={"references": references},
    )


def restore_original_case(record):
    if record.source.get("source_id") != "OVOBENCH":
        raise ValueError("original OVO-Bench record required")
    result = deepcopy(record.task["input"])
    references = deepcopy(record.outcome["references"])
    if "test_info" in result:
        labels = references.pop("test_info")
        for stage, label in zip(result["test_info"], labels, strict=True):
            if set(stage) & set(label):
                raise ValueError("stage inputs overlap scorer references")
            stage.update(label)
    if set(result) & set(references) or "id" in result or "id" in references:
        raise ValueError("case inputs overlap scorer references")
    result.update(references)
    result["id"] = record.task["task_id"]
    if set(result) != set(record.replay["original_field_names"]):
        raise ValueError("original case fields changed")
    return {key: result[key] for key in record.replay["original_field_names"]}
