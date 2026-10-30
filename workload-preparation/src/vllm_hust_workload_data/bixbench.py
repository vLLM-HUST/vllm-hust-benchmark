"""Original scientific question/capsule bindings; no Agent or code execution."""

from copy import deepcopy
from pathlib import Path

from .assets import asset_path, verify_file
from .records import WorkloadRecord
from .registry import source_spec

INPUT_FIELDS = {"question", "data_folder"}
REFERENCE_FIELDS = {"ideal", "distractors", "answer", "hypothesis", "result"}


def capsule_asset(filename):
    for item in source_spec("BIXBENCH")["files"]:
        if item["path"] == filename and item["role"] == "native_environment":
            return {key: item[key] for key in ("path", "sha256", "bytes")}
    raise ValueError("BixBench original registered capsule reference required")


def adapt_case(row, *, source, index):
    required = INPUT_FIELDS | {"question_id", "capsule_uuid", "ideal", "distractors"}
    if (
        source.get("source_id") != "BIXBENCH"
        or not isinstance(row, dict)
        or required - row.keys()
        or not isinstance(row["question_id"], str)
        or not row["question_id"]
        or not isinstance(row["question"], str)
        or not row["question"]
        or not isinstance(row["ideal"], str)
        or not isinstance(row["distractors"], list)
        or row["data_folder"] != f"CapsuleFolder-{row['capsule_uuid']}.zip"
    ):
        raise ValueError("BixBench original flat question/capsule fields required")
    asset = capsule_asset(row["data_folder"])
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": row["question_id"],
            "identity_kind": "original_question_id",
            "native_unit": "scientific_question_with_capsule_environment",
            "input": {key: deepcopy(row[key]) for key in INPUT_FIELDS},
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "assets": [asset],
            "asset_availability": "not_checked_by_metadata_adapter",
            "requirements": [
                "Original scientific Agent, notebook environment and evaluator required; not a ready model request",
                "Verify original capsule before staging; use author preprocessing on a separate working copy to exclude reference notebooks",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "bixbench_original_tasks",
            "source_row_index": index,
            "order": "original_question_file_order_not_request_arrival",
            "annotations": {},
            "synthetic_fallback": False,
        },
        outcome={
            "references": {
                key: deepcopy(row[key]) for key in REFERENCE_FIELDS & row.keys()
            },
            "original_metadata": {
                key: deepcopy(value)
                for key, value in row.items()
                if key not in INPUT_FIELDS | REFERENCE_FIELDS
            },
            "reference_kind": "original_answers_and_source_metadata_not_model_input",
        },
    )


def restore_original_task(record):
    if record.source.get("source_id") != "BIXBENCH":
        raise ValueError("original BixBench record required")
    inputs = record.task["input"]
    references = record.outcome["references"]
    metadata = record.outcome["original_metadata"]
    if (
        set(inputs) != INPUT_FIELDS
        or set(references) - REFERENCE_FIELDS
        or set(metadata) & (INPUT_FIELDS | REFERENCE_FIELDS)
        or metadata.get("question_id") != record.task["task_id"]
    ):
        raise ValueError("original BixBench field/identity separation lost")
    return deepcopy({**inputs, **references, **metadata})


def original_capsule_path(record, *, data_root):
    """Verify and return original ZIP, without extraction or code execution."""
    row = restore_original_task(record)
    asset = capsule_asset(row["data_folder"])
    if record.runtime.get("assets") != [asset]:
        raise ValueError("original BixBench capsule binding changed")
    verify_file(Path(data_root), asset)
    return asset_path(Path(data_root), asset["path"])
