"""Lossless original WebArena-family tasks; evaluator data stays out of inputs."""

from copy import deepcopy
import hashlib
from pathlib import PurePosixPath

from .assets import _json, asset_path, verify_file
from .records import WorkloadRecord

TASKS = "config_files/test.raw.json"
SOURCE_TASKS = {
    "WEBARENA": (TASKS,),
    "VISUALWEBARENA": (
        "config_files/vwa/test_classifieds.raw.json",
        "config_files/vwa/test_reddit.raw.json",
        "config_files/vwa/test_shopping.raw.json",
    ),
}


def pipeline_manifest(source):
    assets = deepcopy(source.get("native_task_assets", source.get("files", [])))
    paths = SOURCE_TASKS.get(source.get("source_id"), ())
    tasks = [entry for name in paths for entry in assets if entry["path"] == name]
    if not paths or not source.get("revision") or len(tasks) != len(paths):
        raise ValueError("pinned original WebArena task array required")
    return {
        **deepcopy(source),
        "unit": "original_web_task_package",
        "provenance_kind": "publisher_original_task_definition",
        "files": [{**entry, "role": "data"} for entry in tasks],
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


def original_task_packages(author_root, *, manifest, task_path=None):
    bound = pipeline_manifest(manifest)
    assets = bound["native_task_assets"]
    if len({entry["path"] for entry in assets}) != len(assets):
        raise ValueError("distinct original WebArena asset paths required")
    for entry in assets:
        verify_file(author_root, entry)
    paths = SOURCE_TASKS[bound["source_id"]]
    if task_path is not None:
        if task_path not in paths:
            raise ValueError("registered original web task file required")
        paths = (task_path,)
    for name in paths:
        entry = next(item for item in assets if item["path"] == name)
        yield from _file_packages(author_root, entry, bound)


def _file_packages(author_root, entry, bound):
    raw = asset_path(author_root, entry["path"]).read_bytes()
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
            visual = bound["source_id"] == "VISUALWEBARENA"
            split = (
                PurePosixPath(entry["path"])
                .name.removeprefix("test_")
                .removesuffix(".raw.json")
            )
            yield {
                "task_id": f"{split}:{row['task_id']}"
                if visual
                else str(row["task_id"]),
                **({"original_split": split} if visual else {}),
                "input": {
                    "original_task": deepcopy(
                        {key: value for key, value in row.items() if key != "eval"}
                    )
                },
                "original_eval": deepcopy(row["eval"]),
                "original_asset": deepcopy(entry),
                "source": {
                    "source_id": bound["source_id"],
                    "revision": bound["revision"],
                },
            }
    finally:
        verify_file(author_root, entry)


def original_rows(path, spec):
    paths = SOURCE_TASKS[spec["original_package_source"]["source_id"]]
    matches = [name for name in paths if path.as_posix().endswith("/" + name)]
    if len(matches) != 1:
        raise ValueError("original WebArena task file required")
    name = matches[0]
    yield from original_task_packages(
        path.parents[len(PurePosixPath(name).parts) - 1],
        manifest=spec["original_package_source"],
        task_path=name,
    )


def adapt_case(row, *, source, index):
    return WorkloadRecord(
        source={**deepcopy(source), **deepcopy(row["source"])},
        task={
            "task_id": row["task_id"],
            "identity_kind": (
                "original_split_and_web_task_id"
                if "original_split" in row
                else "original_web_task_id"
            ),
            "native_unit": "original_web_task_package",
            "input": deepcopy(row["input"]),
            **(
                {"original_split": row["original_split"]}
                if "original_split" in row
                else {}
            ),
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
            "order": (
                "original_split_then_task_array_not_request_arrival"
                if "original_split" in row
                else "original_task_array_not_request_arrival"
            ),
            "synthetic_fallback": False,
            "unmapped_field_names": [],
        },
        outcome={
            "task_id": row["task_id"],
            "original_eval": deepcopy(row["original_eval"]),
            "reference_kind": "original_evaluator_definition_not_observed_result",
        },
    )
