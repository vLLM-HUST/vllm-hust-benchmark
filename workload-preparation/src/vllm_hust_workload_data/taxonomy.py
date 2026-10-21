"""Versioned research classes, separate from historical v1 feature groups."""

from __future__ import annotations

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
