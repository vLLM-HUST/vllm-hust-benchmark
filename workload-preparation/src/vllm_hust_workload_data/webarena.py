"""Lossless original WebArena tasks, with evaluator data kept out of inputs."""

from copy import deepcopy
import hashlib

from .assets import _json, asset_path, verify_file
from .records import WorkloadRecord

TASKS = "config_files/test.raw.json"


def pipeline_manifest(source):
    assets = deepcopy(source.get("native_task_assets", source.get("files", [])))
    tasks = [entry for entry in assets if entry["path"] == TASKS]
    if (
        source.get("source_id") != "WEBARENA"
        or not source.get("revision")
        or len(tasks) != 1
    ):
        raise ValueError("pinned original WebArena task array required")
    return {
        **deepcopy(source),
        "unit": "original_web_task_package",
        "provenance_kind": "publisher_original_task_definition",
        "files": [{**tasks[0], "role": "data"}],
        "native_task_assets": assets,
    }


def relation_spec(source, spec):
    bound = pipeline_manifest(source)
    if bound["revision"] != spec["code_revision"]:
        raise ValueError("WebArena source revision mismatch")
    return {
        **spec,
        "original_package_source": {
            **bound,
            "files": bound["native_task_assets"],
        },
    }


def original_task_packages(author_root, *, manifest):
    bound = pipeline_manifest(manifest)
    assets = bound["native_task_assets"]
    if len({entry["path"] for entry in assets}) != len(assets):
        raise ValueError("distinct original WebArena asset paths required")
    for entry in assets:
        verify_file(author_root, entry)
    entry = next(item for item in assets if item["path"] == TASKS)
    raw = asset_path(author_root, TASKS).read_bytes()
    if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("WebArena task bytes changed during reading")
    rows = _json(raw.decode("utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty original WebArena task array required")
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or type(row.get("task_id")) is not int
            or row["task_id"] < 0
            or row["task_id"] in seen
            or not isinstance(row.get("eval"), dict)
        ):
            raise ValueError(
                "distinct original task IDs and evaluator objects required"
            )
        seen.add(row["task_id"])
    try:
        for row in rows:
            yield {
                "task_id": str(row["task_id"]),
                "input": {
                    "original_task": deepcopy(
                        {key: value for key, value in row.items() if key != "eval"}
                    )
                },
                "original_eval": deepcopy(row["eval"]),
                "original_asset": deepcopy(entry),
                "source": {"source_id": "WEBARENA", "revision": bound["revision"]},
            }
    finally:
        verify_file(author_root, entry)


def original_rows(path, spec):
    if path.name != "test.raw.json" or path.parent.name != "config_files":
        raise ValueError("original WebArena task file required")
    yield from original_task_packages(
        path.parents[1], manifest=spec["original_package_source"]
    )


def adapt_case(row, *, source, index):
    return WorkloadRecord(
        source={**deepcopy(source), **deepcopy(row["source"])},
        task={
            "task_id": row["task_id"],
            "identity_kind": "original_web_task_id",
            "native_unit": "original_web_task_package",
            "input": deepcopy(row["input"]),
        },
        runtime={
            "kind": "interactive_task_package",
            "execution": "not_executed",
            "original_task_asset": deepcopy(row["original_asset"]),
            "requirements": [
                "Original websites, native Agent and original evaluator required; not a model request"
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "webarena_original_packages",
            "source_row_index": index,
            "order": "original_task_array_not_request_arrival",
            "synthetic_fallback": False,
            "unmapped_field_names": [],
        },
        outcome={
            "task_id": row["task_id"],
            "original_eval": deepcopy(row["original_eval"]),
            "reference_kind": "original_evaluator_definition_not_observed_result",
        },
    )
