"""Original publisher task/variation relations; no environment, model, or analysis."""

from copy import deepcopy
import json
from zipfile import ZipFile

from .records import WorkloadRecord


def original_rows(path):
    with ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".json")]
        if len(members) != 1:
            raise ValueError("one original gold-sequence member required")
        payload = json.loads(archive.read(members[0]))
    if not isinstance(payload, dict):
        raise ValueError("original task-key mapping required")
    identities = set()
    for key, task in payload.items():
        if set(task) != {"taskIdx", "taskName", "goldActionSequences"}:
            raise ValueError("unknown original task fields require explicit mapping")
        if (
            type(task["taskIdx"]) is not int
            or str(task["taskIdx"]) != key
            or not isinstance(task["taskName"], str)
        ):
            raise ValueError("original task identity mismatch")
        for offset, case in enumerate(task["goldActionSequences"]):
            if set(case) != {"variationIdx", "fold", "taskDescription", "path"}:
                raise ValueError(
                    "unknown original variation fields require explicit mapping"
                )
            if (
                type(case["variationIdx"]) is not int
                or case["fold"] not in {"train", "dev", "test"}
                or not isinstance(case["taskDescription"], str)
                or not isinstance(case["path"], list)
            ):
                raise ValueError("invalid original variation fields")
            identity = (task["taskIdx"], case["variationIdx"])
            if identity in identities:
                raise ValueError("duplicate original task/variation")
            identities.add(identity)
            yield {
                "taskIdx": task["taskIdx"],
                "taskName": task["taskName"],
                **case,
                "archive_member": members[0],
                "original_task_key": key,
                "original_variation_offset": offset,
            }


def adapt_case(row, *, source, index):
    required = {
        "taskIdx",
        "taskName",
        "variationIdx",
        "taskDescription",
        "fold",
        "path",
        "archive_member",
        "original_task_key",
        "original_variation_offset",
    }
    if required - row.keys():
        raise ValueError("SCIENCEWORLD: missing required fields; no generated fallback")
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": f"{row['taskIdx']}:{row['variationIdx']}",
            "identity_kind": "original_task_and_variation",
            "native_unit": "scienceworld_environment_task",
            "input": {
                key: deepcopy(row[key])
                for key in ["taskIdx", "taskName", "variationIdx", "taskDescription"]
            },
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Original ScienceWorld environment and native agent/scorer; not a ready model request",
                "Publisher gold path is reference-only, not runtime input or a measured model trajectory",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "scienceworld_original_goldpaths",
            "source_row_index": index,
            "order": "original_task_mapping_then_variation_order_not_request_arrival",
            "annotations": {
                key: row[key]
                for key in [
                    "fold",
                    "archive_member",
                    "original_task_key",
                    "original_variation_offset",
                ]
            },
            "unmapped_field_names": [],
            "synthetic_fallback": False,
        },
        outcome={
            "publisher_gold_path": deepcopy(row["path"]),
            "reference_kind": "author_reference_actions_not_observed_model_output",
        },
    )
