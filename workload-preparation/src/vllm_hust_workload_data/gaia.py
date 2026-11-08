"""Original GAIA tasks and attachment relations; no oracle evidence or Agent loop."""

from copy import deepcopy
from pathlib import Path

from .assets import asset_path, iter_rows, verify_file
from .records import WorkloadRecord
from .registry import source_spec

METADATA = {f"2023/{split}/metadata.parquet" for split in ("validation", "test")}
FIELDS = {
    "task_id",
    "Question",
    "Level",
    "Final answer",
    "file_name",
    "file_path",
    "Annotator Metadata",
}


def _identities(values):
    return {x["path"]: (x.get("sha256"), x.get("bytes"), x.get("role")) for x in values}


def relation_spec(manifest, spec):
    if manifest.get("source_id") != "GAIA":
        raise ValueError("original GAIA source identity required")
    files = manifest.get("files", [])
    if len({x["path"] for x in files}) != len(files):
        raise ValueError("duplicate original GAIA asset paths")
    if {x["path"] for x in files if x.get("role") == "data"} != METADATA:
        raise ValueError(
            "original full validation/test metadata required; level views are not extra tasks"
        )
    assets = {x["path"]: deepcopy(x) for x in files}
    if any(not isinstance(x.get("sha256"), str) for x in assets.values()):
        raise ValueError("explicit original GAIA file digests required")
    original = source_spec("GAIA")

    if manifest.get("revision") != original["revision"] or _identities(
        files
    ) != _identities(original["files"]):
        raise ValueError(
            "GAIA manifest must match the pinned original release; no QBI substitute"
        )
    return {**deepcopy(spec), "gaia_original_assets": assets}


def original_rows(path, spec):
    path = Path(path).resolve(strict=True)
    root = path.parents[2]
    relative = path.relative_to(root).as_posix()
    if relative not in METADATA:
        raise ValueError("use full original GAIA metadata, not derived annotated tasks")
    split = path.parent.name
    assets = spec["gaia_original_assets"]
    # Prepared SQLite is also an external input. Re-anchor its stored asset map
    # and metadata identity before reconstructing these protected original tasks.
    if _identities(assets.values()) != _identities(source_spec("GAIA")["files"]):
        raise ValueError("stored GAIA asset identities differ from pinned originals")
    verify_file(root, assets[relative])
    seen = set()
    for row in iter_rows(path):
        if set(row) != FIELDS or any(
            not isinstance(row[k], str) for k in FIELDS - {"Annotator Metadata"}
        ):
            raise ValueError("original GAIA fields/types differ")
        if (
            not row["task_id"]
            or row["task_id"] in seen
            or not row["Question"]
            or row["Level"] not in {"1", "2", "3"}
            or not isinstance(row["Annotator Metadata"], dict)
        ):
            raise ValueError(
                "original GAIA unique ID, question, level and metadata required"
            )
        seen.add(row["task_id"])
        attachment = None
        if row["file_name"]:
            expected = f"2023/{split}/" + row["file_name"]
            if (
                "/" in row["file_name"]
                or "\\" in row["file_name"]
                or Path(row["file_name"]).name != row["file_name"]
                or row["file_path"] != expected
                or expected not in assets
            ):
                raise ValueError(
                    "original GAIA attachment relation is missing or changed"
                )
            attachment = assets[expected]
            verify_file(root, attachment)
            asset_path(root, expected)
        elif row["file_path"]:
            raise ValueError("unbound GAIA attachment path")
        yield {
            "original_case": deepcopy(row),
            "attachment_asset": deepcopy(attachment),
            "split": split,
        }


def adapt_case(row, *, source, index):
    if set(row) != {"original_case", "attachment_asset", "split"}:
        raise ValueError("GAIA missing required fields from original relation reader")
    case = row["original_case"]
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": case["task_id"],
            "identity_kind": "original_gaia_task_id",
            "native_unit": "original_agent_question_with_attachments",
            "input": {
                "question": case["Question"],
                "file_name": case["file_name"],
                "file_path": case["file_path"],
                "attachment_asset": deepcopy(row["attachment_asset"]),
            },
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Run the original question and unmodified attachments using an explicit public Agent/tool profile",
                "No curated local evidence, annotated-task substitute or single-prompt completion claim",
                "Test references are withheld; local validation scoring does not qualify test correctness",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "gaia_original_tasks",
            "source_row_index": index,
            "order": "original_metadata_order_not_request_arrival",
            "annotations": {"level": case["Level"], "split": row["split"]},
            "synthetic_fallback": False,
            "unmapped_field_names": [],
        },
        outcome={
            "reference_answer": case["Final answer"],
            "original_annotator_metadata": deepcopy(case["Annotator Metadata"]),
            "reference_availability": "published_validation"
            if row["split"] == "validation"
            else "withheld_test",
        },
    )


def restore_original_case(record):
    data = record.task["input"]
    return {
        "task_id": record.task["task_id"],
        "Question": data["question"],
        "Level": record.replay["annotations"]["level"],
        "Final answer": record.outcome["reference_answer"],
        "file_name": data["file_name"],
        "file_path": data["file_path"],
        "Annotator Metadata": deepcopy(record.outcome["original_annotator_metadata"]),
    }
