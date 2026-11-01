"""Read unchanged scientific task packages; never execute task or model code.

Problem materials and hidden evaluation materials remain separate. This reader
is preparation infrastructure, not the official Agent/evaluation/judge workflow.
"""

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath

from .assets import asset_path, verify_file
from .registry import read_catalog
from .records import WorkloadRecord


def original_cpu_packages(data_root, *, manifest=None, task_ids=None):
    """Yield one package per original CPU task, verifying every related asset.

    Verification is repeated when the iterator advances or closes. Callers must
    close a partially consumed iterator. No downloads or fallback data are used.
    """
    if manifest is None:
        manifest = read_catalog("naturebench_original_cpu_assets_v1.json")
    if (
        manifest.get("source_id") != "NATUREBENCH"
        or manifest.get("schema_version") != "source-file-manifest/v1"
        or not manifest.get("revision")
        or not manifest.get("dataset_revision")
    ):
        raise ValueError("separate original code and dataset identities required")
    files = {}
    tasks = []
    roles = {
        "problem": "problem_asset",
        "evaluation": "evaluation_asset",
        "environment": "environment_asset",
        "licenses": "license_asset",
    }
    for item in manifest["files"]:
        path = PurePosixPath(item["path"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in item["path"]
            or len(path.parts) < 3
            or path.parts[0] != "tasks"
            or item["path"] in files
        ):
            raise ValueError("distinct contained task assets required")
        if len(path.parts) == 3 and path.name == "metadata.json":
            expected = "data"
            tasks.append(path.parts[1])
        else:
            expected = roles.get(path.parts[2])
        if expected is None or item.get("role") != expected:
            raise ValueError("original asset role or metadata boundary differs")
        files[item["path"]] = item
    if not tasks or {PurePosixPath(name).parts[1] for name in files} != set(tasks):
        raise ValueError("every asset must belong to a registered task metadata")
    if task_ids is not None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or set(task_ids) - set(tasks)
        ):
            raise ValueError("select distinct registered original task IDs")
        tasks = [task for task in tasks if task in task_ids]
    fields = {
        "task_name": str,
        "workflow_topology": str,
        "methodology_paradigm": str,
        "tooling_metadata": type(None),
        "domain_metadata": dict,
        "compute_resource_requirements": dict,
        "performance_entries": list,
    }
    for task_id in tasks:
        prefix = f"tasks/{task_id}/"
        related = [value for name, value in files.items() if name.startswith(prefix)]
        if not set(roles.values()).issubset({item["role"] for item in related}):
            raise ValueError(
                "complete problem, evaluation, environment and license roles required"
            )

        def verify():
            for item in related:
                verify_file(data_root, item)

        verify()
        try:
            metadata = json.loads(
                asset_path(data_root, prefix + "metadata.json").read_bytes()
            )
            if not isinstance(metadata, dict) or set(metadata) != set(fields):
                raise ValueError("original seven-field task metadata required")
            if any(type(metadata[key]) is not kind for key, kind in fields.items()):
                raise ValueError("original task metadata type differs")

            def refs(role):
                return [deepcopy(item) for item in related if item["role"] == role]

            problem = {}
            for name in ("README.md", "data_description.md"):
                relative = prefix + "problem/" + name
                if relative not in files:
                    raise ValueError("original problem description is missing")
                problem[name] = (
                    asset_path(data_root, relative).read_bytes().decode("utf-8")
                )
            yield {
                "task_id": task_id,
                "source": {
                    "source_id": "NATUREBENCH",
                    "repo_id": manifest["repo_id"],
                    "code_revision": manifest["revision"],
                    "dataset_revision": manifest["dataset_revision"],
                },
                "input": {"problem_texts": problem, "assets": refs("problem_asset")},
                "environment_assets": refs("environment_asset"),
                "license_assets": refs("license_asset"),
                "outcome": {
                    "original_task_metadata": metadata,
                    "evaluation_assets": refs("evaluation_asset"),
                },
                "native_execution_complete": False,
            }
        finally:
            verify()


def relation_spec(source, spec):
    """Persist the complete original asset graph with reference-only indexes."""
    if (
        source.get("source_id") != "NATUREBENCH"
        or source.get("revision") != spec["code_revision"]
        or source.get("dataset_revision") != spec["dataset_revision"]
    ):
        raise ValueError("NatureBench original code/data revision mismatch")
    return {
        **spec,
        "original_package_source": {
            "schema_version": "source-file-manifest/v1",
            "source_id": "NATUREBENCH",
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "dataset_revision": source["dataset_revision"],
            "files": deepcopy(source.get("native_task_assets", source["files"])),
        },
    }


def original_rows(path, spec):
    """Resolve a top-level metadata reference to its whole original task package."""
    path = Path(path)
    context = spec["original_package_source"]
    registered = {item["path"] for item in context["files"] if item["role"] == "data"}
    relative = "/".join(path.parts[-3:])
    parts = PurePosixPath(relative).parts
    if relative not in registered or parts[0] != "tasks" or parts[2] != "metadata.json":
        raise ValueError("registered original top-level task metadata required")
    yield from original_cpu_packages(
        path.parents[2], manifest=context, task_ids=[parts[1]]
    )


def adapt_case(row, *, source, index):
    """Expose scientific problem inputs, keeping hidden evaluation separate."""
    return WorkloadRecord(
        source={**deepcopy(source), **deepcopy(row["source"])},
        task={
            "task_id": row["task_id"],
            "identity_kind": "original_scientific_task_directory",
            "native_unit": "scientific_task_package",
            "input": deepcopy(row["input"]),
        },
        runtime={
            "kind": "scientific_task_package",
            "execution": "not_executed",
            "environment_assets": deepcopy(row["environment_assets"]),
            "license_assets": deepcopy(row["license_assets"]),
            "requirements": [
                "Original isolated scientific environment, Agent workflow, evaluator and validity judge required",
                "Not a ready model request; package preparation does not complete native execution",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "naturebench_original_packages",
            "source_row_index": index,
            "order": "original_task_package_not_request_arrival",
            "synthetic_fallback": False,
            "unmapped_field_names": [],
        },
        outcome={
            **deepcopy(row["outcome"]),
            "reference_kind": "original_task_metadata_and_hidden_evaluation_not_observed_output",
        },
    )
