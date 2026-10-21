"""Stable consumer contract for the benchmark-owned preparation distribution."""

from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path

from . import PROVIDER_CONTRACT_VERSION, __version__
from .registry import all_adapters, read_catalog, source_spec
from .taxonomy import research_classes


def provider_identity() -> dict:
    """Check the imported version against distribution metadata, never auto-install."""
    try:
        installed_version = metadata.version("vllm-hust-workload-data")
    except metadata.PackageNotFoundError:
        installed_version = None
    if installed_version is not None and installed_version != __version__:
        raise RuntimeError(
            "a different vllm-hust-workload-data installation shadows this provider"
        )
    try:
        provenance = read_catalog("migration_provenance.json")
    except FileNotFoundError:
        provenance = json.loads(
            (
                Path(__file__).resolve().parents[2] / "MIGRATION_PROVENANCE.json"
            ).read_text(encoding="utf-8")
        )
    return {
        "distribution": "vllm-hust-workload-data",
        "version": __version__,
        "installed_version": installed_version,
        "contract": PROVIDER_CONTRACT_VERSION,
        "module_root": str(Path(__file__).resolve().parent),
        "installation": "INSTALLED"
        if installed_version
        else "SOURCE_ONLY_NOT_INSTALL_VERIFIED",
        "provenance": provenance,
    }


def class_plan(class_id: str) -> dict:
    """Describe a v2 leaf without silently forwarding it as a v1 family filter."""
    classes = {entry["family_id"]: entry for entry in research_classes()}
    if class_id not in classes:
        raise ValueError("unknown workload class")
    definition = classes[class_id]
    rules = [
        rule
        for rule in read_catalog("workload_taxonomy_v2.json")["record_rules"]
        if rule["family_id"] == class_id
    ]
    return {
        "class_id": class_id,
        "compatibility_family": definition["parent_family_id"],
        "candidate_sources": definition["candidate_source_ids"],
        "reviewed_native_subtypes": rules,
        "selection_scope": "CANDIDATE_SOURCES_NOT_RECORD_MEMBERSHIP",
        "instruction": "Select an explicit source and its reviewed native subtype; do not run the whole source as this class.",
    }


def list_sources(*, class_id: str | None = None) -> list[dict]:
    manifests = read_catalog("source_manifests_v1.json")["sources"]
    adapters = all_adapters()
    allowed = (
        set(class_plan(class_id)["candidate_sources"]) if class_id else set(manifests)
    )
    rows = []
    for source_id in sorted(allowed):
        source = source_spec(source_id)
        spec = adapters.get(source_id)
        rows.append(
            {
                "source_id": source_id,
                "revision": source.get("revision"),
                "kind": spec["kind"] if spec else "NOT_IMPLEMENTED",
                "adapter_definition": spec.get("status", "DEFINED")
                if spec
                else "NOT_IMPLEMENTED",
                "pinned_data_files": sum(
                    f.get("role") == "data" for f in source.get("files", [])
                ),
                "local_availability": "NOT_CHECKED_ON_THIS_MACHINE",
                "runtime_validation": "NOT_ESTABLISHED_BY_CATALOG",
                "class_membership": "CANDIDATE_ONLY" if class_id else "NOT_SELECTED",
            }
        )
    return rows
