"""Lossless native MARBLE configurations, not shared model requests."""

from copy import deepcopy
from pathlib import PurePosixPath

from .records import WorkloadRecord

REFERENCE_FIELDS = {"labels", "root_causes", "number_of_labels_pred"}
SCENARIOS = {"bargaining", "coding", "database", "minecraft", "research"}


def adapt_case(row, *, source, index):
    scenario = PurePosixPath(source["file"]).parent.name
    if (
        source.get("source_id") != "MULTIAGENTBENCH"
        or scenario not in SCENARIOS
        or not isinstance(row, dict)
        or not isinstance(row.get("task"), dict)
        or not isinstance(row["task"].get("content"), str)
        or not isinstance(row.get("agents"), list)
        or not isinstance(row.get("relationships"), list)
        or not isinstance(row.get("environment"), dict)
    ):
        raise ValueError("original MARBLE scenario configuration required")
    configuration = deepcopy(row)
    references = {}
    if scenario == "database":
        for key in REFERENCE_FIELDS:
            if key in configuration["task"]:
                references[key] = configuration["task"].pop(key)
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": f"{source['file_sha256']}:{index}",
            "identity_kind": "file_digest_and_row_location_not_invented_original_id",
            "input": deepcopy(configuration["task"]),
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "scenario": scenario,
            "original_configuration": configuration,
            "requirements": [
                "Use original MARBLE Agent/environment and evaluator; not a shared chat prompt",
                "Preserve agent-private configuration and native relationship routing",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "multiagentbench_original_configurations",
            "source_row_index": index,
            "order": "original_file_order_not_request_arrival",
            "synthetic_fallback": False,
        },
        outcome={
            "references": references,
            "reference_kind": "original_database_evaluator_parameters",
        },
    )


def restore_original_configuration(record):
    if record.source.get("source_id") != "MULTIAGENTBENCH":
        raise ValueError("original MARBLE record required")
    configuration = deepcopy(record.runtime["original_configuration"])
    references = record.outcome["references"]
    if (
        configuration["task"] != record.task["input"]
        or set(references) - REFERENCE_FIELDS
        or set(references) & configuration["task"].keys()
        or (references and record.runtime["scenario"] != "database")
    ):
        raise ValueError("native task/reference separation changed")
    configuration["task"].update(deepcopy(references))
    return configuration
