"""Original planning queries and database references; no Agent or research logic."""

from copy import deepcopy
import json
from pathlib import Path

from .records import WorkloadRecord

PROFILES = {
    **{
        f"level_{level}_query_meta.json": (
            "shopping",
            str(level),
            f"database_level{level}.tar.gz",
        )
        for level in (1, 2, 3)
    },
    **{
        f"travelplanning_query_{language}.json": (
            "travel",
            language,
            f"database_{language}.zip",
        )
        for language in ("en", "zh")
    },
}


def original_rows(path):
    path = Path(path)
    if path.name not in PROFILES:
        raise ValueError("unknown original DeepPlanning query file")
    domain, profile, database = PROFILES[path.name]
    rows = json.loads(path.read_text("utf-8"))
    if not isinstance(rows, list):
        raise ValueError("original task array required")
    seen = set()
    expected = (
        {"id", "query"}
        if domain == "shopping"
        else {"id", "query", "meta_info", "query_with_constraints"}
    )
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != expected:
            raise ValueError("unknown original fields require explicit mapping")
        if type(row["id"]) not in (int, str) or not isinstance(row["query"], str):
            raise ValueError("original ID/query types required")
        if str(row["id"]) in seen:
            raise ValueError("duplicate original task ID")
        seen.add(str(row["id"]))
        yield {
            "original": row,
            "domain": domain,
            "profile": profile,
            "database_archive": "databases/" + database,
            "original_query_file": path.name,
            "original_query_offset": index,
        }


def adapt_case(row, *, source, index):
    required = {
        "original",
        "domain",
        "profile",
        "database_archive",
        "original_query_file",
        "original_query_offset",
    }
    if not isinstance(row, dict) or required - row.keys():
        raise ValueError("DEEPPLANNING: missing required fields; no generated fallback")
    original = deepcopy(row["original"])
    domain = row["domain"]
    prefix = "case_" if domain == "shopping" else "id_"
    return WorkloadRecord(
        source=deepcopy(source),
        task={
            "task_id": f"{domain}:{row['profile']}:{original['id']}",
            "identity_kind": "original_domain_profile_task_id",
            "native_unit": "deepplanning_environment_task",
            "input": {"id": original["id"], "query": original["query"]},
        },
        runtime={
            "kind": "native_environment",
            "execution": "not_executed",
            "requirements": [
                "Original per-task database, native tool/Agent loop and domain scorer required; not a ready model request",
                "Travel includes original report-conversion stage; preserve absent original transport tables and native no-information handling",
            ],
        },
        replay={
            "adapter_version": "source-adapter/v1",
            "mapping": "deepplanning_original_tasks",
            "source_row_index": index,
            "order": "original_file_and_task_order_not_request_arrival",
            "annotations": {
                key: row[key]
                for key in (
                    "domain",
                    "profile",
                    "database_archive",
                    "original_query_file",
                    "original_query_offset",
                )
            },
            "native_database_case": prefix + str(original["id"]),
            "unmapped_field_names": [],
            "synthetic_fallback": False,
        },
        outcome={
            "original_evaluation_metadata": {
                key: value
                for key, value in original.items()
                if key not in {"id", "query"}
            },
            "reference_kind": "original_task_metadata_and_external_database_validation_not_observed_output",
        },
    )
