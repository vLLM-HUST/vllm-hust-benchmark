"""Lossless references to original AppWorld task bundles, never flattened QA.

Reads existing private publisher data only. Does not import task evaluation code,
execute generated code, download anything, or construct replacement prompts.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from .records import WorkloadRecord

SPLITS = ("train", "dev", "test_normal", "test_challenge")


def file_reference(path, root):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("original task asset missing or escapes private data root")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def original_task_bundles(data_root):
    root = Path(data_root).resolve(strict=True)
    shared = {}
    for name in ("base_dbs", "api_docs"):
        shared[name] = [
            file_reference(path, root)
            for path in sorted((root / name).rglob("*"))
            if path.is_file()
        ]
        if not shared[name]:
            raise ValueError("original shared base databases and API docs required")
    selected = []
    seen = set()
    for split in SPLITS:
        split_path = root / "datasets" / f"{split}.txt"
        split_reference = file_reference(split_path, root)
        identities = split_path.read_text(encoding="utf-8").splitlines()
        if not identities:
            raise ValueError("empty original split")
        for offset, identity in enumerate(identities):
            if not re.fullmatch(r"[A-Za-z0-9_-]+", identity) or identity in seen:
                raise ValueError("invalid, duplicate or overlapping original task ID")
            seen.add(identity)
            selected.append((split, offset, identity, split_reference))
    for split, offset, identity, split_reference in selected:
        folder = root / "tasks" / identity
        spec_path = folder / "specs.json"
        spec_reference = file_reference(spec_path, root)
        specs = json.loads(spec_path.read_text(encoding="utf-8"))
        if (
            not isinstance(specs, dict)
            or not isinstance(specs.get("instruction"), str)
            or not specs["instruction"]
        ):
            raise ValueError("original task specs/instruction required")
        groups = {}
        for name in ("dbs", "ground_truth"):
            files = sorted((folder / name).rglob("*"))
            groups[name] = [
                file_reference(path, root) for path in files if path.is_file()
            ]
            if not groups[name]:
                raise ValueError("original database and reference bundles required")
        yield {
            "shared_environment_files": deepcopy(shared),
            "task_id": identity,
            "split": split,
            "split_offset": offset,
            "original_specs": specs,
            "spec_file": spec_reference,
            "split_file": split_reference,
            "database_files": groups["dbs"],
            "reference_files": groups["ground_truth"],
        }


def adapt_case(row, *, source, index):
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": row["task_id"],
            "identity_kind": "original_appworld_task_id",
            "native_unit": "appworld_application_environment_task",
            "input": {
                "original_specs": deepcopy(row["original_specs"]),
                "database_files": deepcopy(row["database_files"]),
                "shared_environment_files": deepcopy(row["shared_environment_files"]),
            },
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Original AppWorld data root including shared base_dbs and api_docs required",
                "Original Agent/environment/evaluator required; not a ready model request",
                "Reference files must never be exposed as model input",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "appworld_original_task_bundles",
            "source_row_index": index,
            "order": "original_split_file_order_not_request_arrival",
            "annotations": {
                key: deepcopy(row[key])
                for key in ("split", "split_offset", "spec_file", "split_file")
            },
            "unmapped_field_names": [],
            "synthetic_fallback": False,
        },
        outcome={
            "reference_files": deepcopy(row["reference_files"]),
            "reference_kind": "original_private_evaluator_and_answer_assets_not_model_input",
        },
    )
