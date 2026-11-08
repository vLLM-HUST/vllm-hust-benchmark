"""Preserve original SWE-smith tasks without exposing bug patches to an Agent."""

from copy import deepcopy

from .records import WorkloadRecord

INPUT_FIELDS = {"problem_statement", "repo", "image_name"}
REFERENCE_FIELDS = {"patch", "FAIL_TO_PASS", "PASS_TO_PASS"}
FIELDS = INPUT_FIELDS | REFERENCE_FIELDS | {"instance_id"}


def adapt_case(row, *, source, index):
    if (
        source.get("source_id") != "SWESMITH"
        or not isinstance(row, dict)
        or set(row) != FIELDS
        or any(
            not isinstance(row[name], str)
            for name in INPUT_FIELDS | {"instance_id", "patch"}
        )
        or any(not row[name] for name in ("instance_id", "repo", "image_name"))
        or any(
            not isinstance(row[name], list)
            or any(not isinstance(test, str) for test in row[name])
            for name in ("FAIL_TO_PASS", "PASS_TO_PASS")
        )
    ):
        raise ValueError(
            "original SWE-smith seven-field task required; payload suppressed"
        )
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": row["instance_id"],
            "identity_kind": "original_instance_id",
            "native_unit": "repository_bug_repair_task",
            "input": {name: deepcopy(row[name]) for name in INPUT_FIELDS},
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Use original repository instance checkout, Agent and official scoring",
                "Original patch introduces a bug; gold mode reverses it, candidate repair applies forward",
                "Empty original issue is preserved, not replaced with an invented prompt",
                "Original official container paths request x86_64; runtime compatibility must be verified separately",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "swesmith_original_tasks",
            "source_row_index": index,
            "order": "original_shard_row_order_not_request_arrival",
            "synthetic_fallback": False,
        },
        outcome={
            "references": {name: deepcopy(row[name]) for name in REFERENCE_FIELDS},
            "reference_kind": "original_bug_patch_and_test_partitions_not_model_input",
        },
    )


def restore_original_task(record):
    if record.source.get("source_id") != "SWESMITH":
        raise ValueError("original SWE-smith record required")
    inputs = record.task["input"]
    references = record.outcome["references"]
    if set(inputs) != INPUT_FIELDS or set(references) != REFERENCE_FIELDS:
        raise ValueError("original SWE-smith input/reference separation lost")
    return deepcopy({"instance_id": record.task["task_id"], **inputs, **references})
