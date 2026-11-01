"""Lossless original offline web tasks; labels are not model request inputs."""

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import zipfile

from .assets import iter_rows
from .records import WorkloadRecord

TASK_FIELDS = {"annotation_id", "website", "domain", "subdomain", "confirmed_task"}
PAGE_FIELDS = {"action_uid", "raw_html", "cleaned_html"}


def original_rows(path):
    path = Path(path)
    if path.suffix.lower() != ".zip":
        yield from iter_rows(path)
        return
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError("unsafe Mind2Web archive member")
            if not name.endswith(".json"):
                continue
            if member.parent.as_posix() not in {
                "test_task",
                "test_website",
                "test_domain",
            }:
                raise ValueError("unexpected original Mind2Web split")
            rows = json.loads(archive.read(name, pwd=b"mind2web"))
            if not isinstance(rows, list):
                raise ValueError("original task array required")
            yield from rows


def adapt_case(row, *, source, index):
    if (
        not TASK_FIELDS <= row.keys()
        or not isinstance(row.get("actions"), list)
        or not row["actions"]
    ):
        raise ValueError("complete original Mind2Web task required")
    if not isinstance(row.get("action_reprs"), list) or len(row["action_reprs"]) != len(
        row["actions"]
    ):
        raise ValueError("Mind2Web original action relation mismatch")
    pages, references, seen = [], [], set()
    for action in row["actions"]:
        if not PAGE_FIELDS <= action.keys() or action["action_uid"] in seen:
            raise ValueError("Mind2Web original page or action identity invalid")
        seen.add(action["action_uid"])
        pages.append({key: deepcopy(action[key]) for key in PAGE_FIELDS})
        references.append(
            {
                key: deepcopy(value)
                for key, value in action.items()
                if key not in PAGE_FIELDS
            }
        )
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": row["annotation_id"],
            "identity_kind": "original_annotation_id",
            "native_unit": "complete_original_offline_web_task",
            "input": {
                **{key: deepcopy(row[key]) for key in TASK_FIELDS},
                "pages": pages,
            },
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Original candidate ranking, reference prior-action protocol and author evaluator required; not a live browser or ready model request"
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "mind2web_original_tasks",
            "source_row_index": index,
            "order": "original_action_order_not_request_arrival",
            "annotations": {},
            "unmapped_field_names": [],
            "synthetic_fallback": False,
        },
        outcome={
            "reference_actions": references,
            "action_reprs": deepcopy(row["action_reprs"]),
            "original_task_extensions": {
                key: deepcopy(value)
                for key, value in row.items()
                if key not in TASK_FIELDS | {"actions", "action_reprs"}
            },
            "reference_kind": "original_demonstrated_operations_and_candidate_labels_not_model_outputs",
        },
    )


def restore_original_task(record):
    inputs = deepcopy(record.task["input"])
    pages = inputs.pop("pages")
    references = record.outcome["reference_actions"]
    if len(pages) != len(references):
        raise ValueError("Mind2Web original action alignment lost")
    return {
        **inputs,
        **deepcopy(record.outcome["original_task_extensions"]),
        "actions": [
            {**page, **deepcopy(reference)}
            for page, reference in zip(pages, references, strict=True)
        ],
        "action_reprs": deepcopy(record.outcome["action_reprs"]),
    }
