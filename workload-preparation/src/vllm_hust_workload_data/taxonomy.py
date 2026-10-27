"""Versioned research classes, separate from historical v1 feature groups."""

from __future__ import annotations

from pathlib import PurePosixPath

from .registry import read_catalog

EVIDENCE_KINDS = frozenset(
    {
        "task_definition",
        "model_trace",
        "human_demonstration",
        "synthetic_dialogue",
        "historical_environment_trace",
    }
)


def research_classes(root=None):
    base = read_catalog("workload_taxonomy_v1.json", root)["canonical_families"]
    refined = read_catalog("workload_taxonomy_v2.json", root)
    parents = {r["family_id"]: r for r in base}
    replaced = {r["parent_family_id"] for r in refined["replacements"]}
    if not replaced <= parents.keys():
        raise ValueError("unknown parent feature group")
    rows = []
    for parent in base:
        children = [
            r
            for r in refined["replacements"]
            if r["parent_family_id"] == parent["family_id"]
        ]
        if children:
            rows.extend(
                {**r, "taxonomy_version": refined["taxonomy_version"]} for r in children
            )
        else:
            rows.append(
                {
                    **parent,
                    "parent_family_id": parent["family_id"],
                    "taxonomy_version": refined["taxonomy_version"],
                    "status": "RETAINED_DEFINITION_RUNTIME_GENERALIZATION_UNVALIDATED",
                    "exclusion": parent["exclusion_and_evidence_boundary"],
                    "system_measurements": [],
                }
            )
    ids = [r["family_id"] for r in rows]
    if len(ids) != len(set(ids)) or replaced & set(ids):
        raise ValueError("duplicate class or replaced parent retained as leaf")
    sources = {
        r["source_id"]
        for r in read_catalog("authoritative_datasets_v1.json", root)["sources"]
    }
    if any(not set(r["candidate_source_ids"]) <= sources for r in rows):
        raise ValueError("unknown candidate source")
    keys = set()
    for rule in refined["record_rules"]:
        key = (rule["source_id"], rule["native_subtype"])
        if (
            key in keys
            or rule["family_id"] not in ids
            or rule["source_id"] not in sources
        ):
            raise ValueError("ambiguous or invalid record rule")
        keys.add(key)
    return rows


def delivery_classes(root=None):
    """Explicit 40-class plan, not a replacement for the research default."""
    plan = read_catalog("workload_delivery_classification_v1.json", root)
    rows = plan["classes"]
    legacy = {row["family_id"] for row in research_classes(root)}
    additions = {"constraint-planning", "text-to-speech-generation"}
    if (
        plan["schema_version"] != "workload-delivery-classification/v1"
        or plan["selection_scope"] != "CANDIDATE_SOURCES_NOT_RECORD_MEMBERSHIP"
        or len(rows) != 40
        or [row["workload_id"] for row in rows] != [f"W{i:02}" for i in range(1, 41)]
        or {row["family_id"] for row in rows} != legacy | additions
    ):
        raise ValueError(
            "40-class plan must preserve existing IDs and only declared additions"
        )
    registered = set(read_catalog("source_manifests_v1.json", root)["sources"])
    result = []
    for row in rows:
        candidates = row["candidate_source_ids"]
        if (
            not candidates
            or len(set(candidates)) != len(candidates)
            or any(not isinstance(sid, str) or not sid for sid in candidates)
            or row["native_validation"] != "NOT_ESTABLISHED_BY_CLASSIFICATION"
        ):
            raise ValueError(
                "explicit candidates without native readiness claims required"
            )
        expected = (
            "PLANNED_ADDITIONAL_CLASS_ONLY"
            if row["family_id"] in additions
            else "EXISTING_RESEARCH_CLASS_DEFINITION"
        )
        if row["definition_status"] != expected:
            raise ValueError(
                "planned additions cannot impersonate existing research classes"
            )
        result.append(
            {
                **row,
                "selection_scope": plan["selection_scope"],
                "candidate_provider_registration": [
                    {
                        "source_id": sid,
                        "data_provider_source_registered": sid in registered,
                        "other_benchmark_entrypoints_checked": False,
                    }
                    for sid in candidates
                ],
            }
        )
    return result


def delivery_assets(root=None):
    """All plan sources, including traces and private request assets, no raw data."""
    classes = delivery_classes(root)
    plan = read_catalog("workload_delivery_classification_v1.json", root)
    registered = set(read_catalog("source_manifests_v1.json", root)["sources"])
    result = {}
    for row in classes:
        for source_id in row["candidate_source_ids"]:
            item = result.setdefault(
                source_id,
                {
                    "source_id": source_id,
                    "asset_group": "task-class-candidates",
                    "related_workload_ids": [],
                    "selection_scope": plan["selection_scope"],
                },
            )
            item["related_workload_ids"].append(row["workload_id"])
    class_ids = {r["workload_id"] for r in classes}
    groups = {
        "serving-traces",
        "cross-task-relationships",
        "controlled-long-context-probes",
        "enterprise-private-requests",
    }
    for row in plan["supplemental_assets"]:
        source_id = row["source_id"]
        if (
            not isinstance(source_id, str)
            or not source_id
            or source_id in result
            or row["asset_group"] not in groups
            or not set(row["related_workload_ids"]) <= class_ids
            or len(row["related_workload_ids"]) != len(set(row["related_workload_ids"]))
            or not isinstance(row["boundary"], str)
            or not row["boundary"].strip()
        ):
            raise ValueError(
                "unique supplemental asset with explicit boundary required"
            )
        result[source_id] = {
            **row,
            "selection_scope": "ASSET_RELATION_NOT_RECORD_MEMBERSHIP",
        }
    if len(result) != 118:
        raise ValueError("complete fixed 118-source delivery classification required")
    entrypoints = plan.get("source_entrypoints", {})
    if not set(entrypoints) <= set(result):
        raise ValueError("entrypoint index refers to unknown plan source")
    roles = {
        "original_profile",
        "original_request_replay",
        "data_reading_only",
        "generation",
        "prepare_scoring_inputs",
        "isolated_scoring",
        "prepare_assets",
        "saved_output_reuse_audit",
    }
    for entries in entrypoints.values():
        seen = set()
        for entry in entries:
            path = PurePosixPath(entry["path"])
            key = (entry["role"], entry["path"])
            if (
                entry["role"] not in roles
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in entry["path"]
                or ":" in entry["path"]
                or key in seen
            ):
                raise ValueError(
                    "distinct repository-relative execution metadata required"
                )
            seen.add(key)
    return [
        {
            **result[source_id],
            "data_provider_source_registered": source_id in registered,
            "native_validation": "NOT_ESTABLISHED_BY_CLASSIFICATION",
            "local_availability": "NOT_CHECKED_ON_THIS_MACHINE",
            "entrypoints": entrypoints.get(source_id, []),
            "entrypoint_distribution": "BENCHMARK_REPOSITORY_PATHS_NOT_EXECUTORS_IN_DATA_WHEEL",
            "entrypoint_scope": "DECLARED_SCRIPT_SCOPE_ONLY_NOT_FULL_SOURCE_COVERAGE; consult protocol/environment docs and the separate delivery ledger",
            "entrypoint_validation": "NOT_ESTABLISHED_BY_INDEX; empty list is not proof of missing implementation",
        }
        for source_id in sorted(result)
    ]


def classify_record(source_id, native_subtype, *, evidence_kind, root=None):
    """Match an explicitly reviewed native subtype; never infer from source alone.

    The caller must obtain subtype from the task contract/source fields, not
    outcomes or generated answers. A match classifies task type, not runtime proof.
    """
    if evidence_kind not in EVIDENCE_KINDS:
        raise ValueError("explicit evidence kind required")
    classes = {r["family_id"]: r for r in research_classes(root)}
    rules = read_catalog("workload_taxonomy_v2.json", root)["record_rules"]
    rule = next(
        (
            r
            for r in rules
            if r["source_id"] == source_id and r["native_subtype"] == native_subtype
        ),
        None,
    )
    return {
        "taxonomy_version": "workload-taxonomy/v2",
        "source_id": source_id,
        "native_subtype": native_subtype,
        "evidence_kind": evidence_kind,
        "family_id": rule["family_id"] if rule else None,
        "parent_family_id": classes[rule["family_id"]]["parent_family_id"]
        if rule
        else None,
        "status": "MATCHED_REVIEWED_SUBTYPE"
        if rule
        else "UNCLASSIFIED_REVIEW_REQUIRED",
        "runtime_generalization": "NOT_ESTABLISHED_BY_CLASSIFICATION",
    }
