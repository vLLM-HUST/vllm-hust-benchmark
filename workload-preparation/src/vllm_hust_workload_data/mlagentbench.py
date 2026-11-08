"""Original registry aliases and native directory tasks, never invented prompts."""

from copy import deepcopy
import json
from pathlib import Path
import re

from .assets import verify_file
from .records import WorkloadRecord
from .registry import source_spec

REGISTRY = "MLAgentBench/benchmarks/tasks.json"


def relation_spec(manifest, spec):
    return {**deepcopy(spec), "original_mlagentbench_source": deepcopy(manifest)}


def original_rows(path, spec):
    path = Path(path).resolve(strict=True)
    directory_task = (
        path.name == "research_problem.txt" and path.parent.name == "scripts"
    )
    root = path.parents[4] if directory_task else path.parents[2]
    relative = path.relative_to(root).as_posix()
    if relative != REGISTRY and (
        not directory_task or not relative.startswith("MLAgentBench/benchmarks/")
    ):
        raise ValueError(
            "original author registry or native task-definition path required"
        )
    manifest = spec.get("original_mlagentbench_source") or source_spec("MLAGENTBENCH")
    if manifest["source_id"] != "MLAGENTBENCH" or not manifest.get("revision"):
        raise ValueError("pinned original MLAgentBench source manifest required")
    files = manifest["files"]
    names = {item["path"] for item in files}
    if len(names) != len(files) or REGISTRY not in names:
        raise ValueError("distinct original software and task registry required")

    def verify():
        for item in files:
            verify_file(root, item)

    verify()
    text = (root / REGISTRY).read_bytes().decode("utf-8")
    cases = json.loads(text)
    if not isinstance(cases, dict) or not cases:
        raise ValueError("original nonempty task-registry object required")
    definition_text = path.read_bytes().decode("utf-8")
    if directory_task:
        folder = path.parents[1].name
        if folder in cases:
            raise ValueError(
                "registered alias overrides native folder; select original registry"
            )
        # This is exactly get_task_info(folder)'s published filesystem branch,
        # not a new registry entry or a reconstructed prompt.
        cases = {
            folder: {
                "benchmark_folder_name": folder,
                "research_problem": definition_text,
            }
        }
    try:
        for case_id, case in cases.items():
            if (
                not isinstance(case_id, str)
                or not case_id
                or not isinstance(case, dict)
            ):
                raise ValueError("original named task-registry case required")
            folder = case.get("benchmark_folder_name")
            problem = case.get("research_problem")
            if not isinstance(folder, str) or not re.fullmatch(
                r"[A-Za-z0-9_-]+", folder
            ):
                raise ValueError(
                    "original contained scientific benchmark folder required"
                )
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(
                    "explicit original registered research problem required"
                )
            prefix = "MLAgentBench/benchmarks/" + folder + "/"
            related = [
                deepcopy(item) for item in files if item["path"].startswith(prefix)
            ]
            if not related or not any(
                item["path"].startswith(prefix + "env/") for item in related
            ):
                raise ValueError("original scientific environment relation missing")
            evaluation = [
                item for item in related if item["role"] == "evaluation_asset"
            ]
            if not evaluation and not directory_task:
                raise ValueError("original evaluation software relation missing")
            yield {
                "case_id": case_id,
                "benchmark_folder_name": folder,
                "research_problem": problem,
                "original_registry_case": None if directory_task else deepcopy(case),
                "original_registry_text": None if directory_task else text,
                "definition_origin": "native_directory"
                if directory_task
                else "author_registry",
                "definition_file": relative,
                "original_definition_text": definition_text,
                "task_software": [
                    item for item in related if item["role"] != "evaluation_asset"
                ],
                "evaluation_assets": evaluation,
                "license_assets": [
                    deepcopy(item) for item in files if item["role"] == "license_asset"
                ],
            }
    finally:
        verify()


def adapt_case(row, *, source, index):
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": row["case_id"],
            "identity_kind": "original_native_directory_name"
            if row["definition_origin"] == "native_directory"
            else "original_registered_case_id",
            "native_unit": "scientific_task_definition",
            "input": {
                "research_problem": row["research_problem"],
                "benchmark_folder_name": row["benchmark_folder_name"],
            },
        },
        runtime={
            "kind": "scientific_task_package",
            "execution": "not_executed",
            "task_software": deepcopy(row["task_software"]),
            "license_assets": deepcopy(row["license_assets"]),
            "underlying_scientific_data_verified": False,
            "requirements": [
                "Original isolated ResearchAgent/environment and evaluator required; not a ready model request",
                "Registry aliases and published directory-task definitions retain their distinct origins; directory count is not paper benchmark count",
                "Original scientific datasets, access conditions and selected native protocol are verified separately",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "mlagentbench_original_registry",
            "source_row_index": index,
            "order": "native_definition_file_not_request_arrival"
            if row["definition_origin"] == "native_directory"
            else "original_registry_case_order_not_request_arrival",
            "original_registry_case": deepcopy(row["original_registry_case"]),
            "original_registry_text": row["original_registry_text"],
            "definition_origin": row["definition_origin"],
            "definition_file": row["definition_file"],
            "original_definition_text": row["original_definition_text"],
            "synthetic_fallback": False,
            "unmapped_field_names": [],
        },
        outcome={
            "evaluation_assets": deepcopy(row["evaluation_assets"]),
            "original_evaluator_available": bool(row["evaluation_assets"]),
            "reference_kind": "original_evaluator_software_not_an_observed_score",
        },
    )
