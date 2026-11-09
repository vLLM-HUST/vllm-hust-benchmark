"""Preserve original paper packages and hidden grading assets; never render prompts."""

from copy import deepcopy
from pathlib import Path, PurePosixPath

from .assets import asset_path, verify_file
from .records import WorkloadRecord
from .registry import read_catalog

PREFIX = ("project", "paperbench", "data", "papers")


def asset_role(name):
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or path.as_posix() != name
        or path.parts[:4] != PREFIX
        or len(path.parts) < 6
    ):
        raise ValueError("contained original paper asset required")
    leaf = path.parts[5]
    if len(path.parts) == 6:
        if leaf == "config.yaml":
            return "data"
        if leaf in {"rubric.json", "judge.addendum.md"}:
            return "evaluation_asset"
        if leaf in {"paper.md", "paper.pdf", "addendum.md", "blacklist.txt"}:
            return "problem_asset"
    elif leaf in {"assets", "judge"}:
        return "problem_asset" if leaf == "assets" else "evaluation_asset"
    raise ValueError("unreviewed original paper asset role")


def original_paper_packages(data_root, *, manifest=None, task_ids=None):
    """Read immutable source packages, not Agent requests or task execution results."""
    if manifest is None:
        manifest = read_catalog("paperbench_original_paper_assets_v1.json")
    if (
        manifest.get("source_id") != "PAPERBENCH"
        or manifest.get("schema_version") != "source-file-manifest/v1"
        or manifest.get("repo_id") != "openai/frontier-evals"
        or not manifest.get("revision")
    ):
        raise ValueError("explicit original PaperBench identity required")
    files = {}
    tasks = []
    for item in manifest["files"]:
        name = item["path"]
        if name in files or item.get("role") != asset_role(name):
            raise ValueError(
                "distinct original assets with unchanged input/evaluation roles required"
            )
        files[name] = item
        if item["role"] == "data":
            tasks.append(PurePosixPath(name).parts[4])
    if not tasks or {PurePosixPath(name).parts[4] for name in files} != set(tasks):
        raise ValueError("every asset requires original paper configuration")
    if task_ids is not None:
        if (
            not task_ids
            or len(set(task_ids)) != len(task_ids)
            or set(task_ids) - set(tasks)
        ):
            raise ValueError("select distinct original paper directory identities")
        tasks = [task for task in tasks if task in task_ids]
    for task_id in sorted(tasks):
        prefix = "/".join((*PREFIX, task_id)) + "/"
        related = [x for name, x in files.items() if name.startswith(prefix)]
        required = {
            prefix + name
            for name in [
                "config.yaml",
                "paper.md",
                "paper.pdf",
                "addendum.md",
                "blacklist.txt",
                "rubric.json",
            ]
        }
        if not required <= set(files):
            raise ValueError(
                "complete original configuration,paper,instructions and rubric required"
            )

        def verify():
            for item in related:
                verify_file(data_root, item)

        verify()
        try:
            config = files[prefix + "config.yaml"]
            yield {
                "task_id": task_id,
                "source": {
                    "source_id": "PAPERBENCH",
                    "repo_id": manifest["repo_id"],
                    "revision": manifest["revision"],
                },
                "input": {
                    "original_config_yaml": asset_path(data_root, config["path"])
                    .read_bytes()
                    .decode("utf-8"),
                    "config_asset": deepcopy(config),
                    "paper_assets": [
                        deepcopy(x) for x in related if x["role"] == "problem_asset"
                    ],
                },
                "outcome": {
                    "evaluation_assets": [
                        deepcopy(x) for x in related if x["role"] == "evaluation_asset"
                    ]
                },
                "native_execution_complete": False,
            }
        finally:
            verify()


def relation_spec(source, spec):
    if (
        source.get("source_id") != "PAPERBENCH"
        or source.get("revision") != spec["code_revision"]
    ):
        raise ValueError("PaperBench original code/data revision mismatch")
    return {
        **spec,
        "original_package_source": {
            "schema_version": "source-file-manifest/v1",
            "source_id": "PAPERBENCH",
            "repo_id": source["repo_id"],
            "revision": source["revision"],
            "files": deepcopy(source.get("native_task_assets", source["files"])),
        },
    }


def original_rows(path, spec):
    path = Path(path)
    context = spec["original_package_source"]
    registered = {x["path"] for x in context["files"] if x["role"] == "data"}
    relative = "/".join(path.parts[-6:])
    if relative not in registered or asset_role(relative) != "data":
        raise ValueError("registered original paper config reference required")
    yield from original_paper_packages(
        path.parents[5], manifest=context, task_ids=[PurePosixPath(relative).parts[4]]
    )


def adapt_case(row, *, source, index):
    return WorkloadRecord(
        source={**deepcopy(source), **deepcopy(row["source"])},
        task={
            "task_id": row["task_id"],
            "identity_kind": "original_paper_directory",
            "native_unit": "scientific_paper_package",
            "input": deepcopy(row["input"]),
        },
        runtime={
            "kind": "scientific_paper_package",
            "execution": "not_executed",
            "native_entry": "paperbench.nano.entrypoint",
            "requirements": [
                "Original Agent rollout,reproduction and rubric grading remain required",
                "Material references are not a ready model request or observed task result",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "paperbench_original_packages",
            "source_row_index": index,
            "order": "original_paper_directory_not_arrival_trace",
            "synthetic_fallback": False,
            "unmapped_field_names": [],
        },
        outcome={
            **deepcopy(row["outcome"]),
            "reference_kind": "original_rubric_and_judge_assets_not_observed_output",
        },
    )
