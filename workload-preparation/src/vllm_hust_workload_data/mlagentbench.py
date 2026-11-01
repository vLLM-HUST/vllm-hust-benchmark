"""Original author task-registry cases and their scientific software relations."""

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
    root = path.parents[2]
    if path.relative_to(root).as_posix() != REGISTRY:
        raise ValueError("original author MLAgentBench task registry path required")
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
    text = path.read_bytes().decode("utf-8")
    cases = json.loads(text)
    if not isinstance(cases, dict) or not cases:
        raise ValueError("original nonempty task-registry object required")
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
            if not evaluation:
                raise ValueError("original evaluation software relation missing")
            yield {
                "case_id": case_id,
                "benchmark_folder_name": folder,
                "research_problem": problem,
                "original_registry_case": deepcopy(case),
                "original_registry_text": text,
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
            "identity_kind": "original_registered_case_id",
            "native_unit": "scientific_task_registry_case",
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
                "Author task registry only; additional native folder task names are not invented registry rows",
                "Original scientific datasets, access conditions and selected native protocol are verified separately",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "mlagentbench_original_registry",
            "source_row_index": index,
            "order": "original_registry_case_order_not_request_arrival",
            "original_registry_case": deepcopy(row["original_registry_case"]),
            "original_registry_text": row["original_registry_text"],
            "synthetic_fallback": False,
            "unmapped_field_names": [],
        },
        outcome={
            "evaluation_assets": deepcopy(row["evaluation_assets"]),
            "reference_kind": "original_evaluator_software_not_an_observed_score",
        },
    )
