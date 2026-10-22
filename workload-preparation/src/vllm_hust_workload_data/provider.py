"""Stable consumer contract for the benchmark-owned preparation distribution."""

from __future__ import annotations

from importlib import metadata
import json
import hashlib
from pathlib import Path
import zipfile

from . import PROVIDER_CONTRACT_VERSION, __version__
from .registry import all_adapters, read_catalog, source_spec
from .taxonomy import research_classes

TAU2_ORIGINAL_REVISION = "f8de30c298689cbe0117d76a378e7315a17e5bd8"  # pragma: allowlist secret - public upstream revision
TAU2_ORIGINAL_ARCHIVE_SHA256 = "65f1d59b3783bfb38664142fb166aa40b369e4b4083170723059d8f1883ccd64"  # pragma: allowlist secret - official archive checksum


def tau2_task_references(*, official_archive, data_root, task_ids):
    """Return original telecom task references, never user/scoring payloads.

    This is preparation only. The official runner owns task loading, user
    simulation, environment operations and scoring. Caller fixes IDs first.
    """
    archive_path = Path(official_archive)
    if (
        hashlib.sha256(archive_path.read_bytes()).hexdigest()
        != TAU2_ORIGINAL_ARCHIVE_SHA256
    ):
        raise ValueError("official tau2 archive identity mismatch")
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or any(not isinstance(value, str) or not value for value in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("explicit unique original task IDs required")
    relative = "tau2/domains/telecom/tasks_full.json"
    path = Path(data_root) / relative
    raw = path.read_bytes()
    with zipfile.ZipFile(archive_path) as archive:
        names = [
            name for name in archive.namelist() if name.endswith("/data/" + relative)
        ]
        if len(names) != 1 or raw != archive.read(names[0]):
            raise ValueError("native task file differs from official archive")
    rows = json.loads(raw)
    positions = {row["id"]: index for index, row in enumerate(rows)}
    if len(positions) != len(rows) or not set(task_ids) <= positions.keys():
        raise ValueError("unknown or ambiguous original task ID")
    digest = hashlib.sha256(raw).hexdigest()
    return [
        {
            "schema_version": "native-task-reference/v1",
            "source_id": "TAUBENCH",
            "revision": TAU2_ORIGINAL_REVISION,
            "domain": "telecom",
            "task_set": "telecom_full",
            "native_task_id": identifier,
            "input_ref": {
                "file": relative,
                "sha256": digest,
                "row_index": positions[identifier],
            },
            "runtime_kind": "official_native_task",
            "scoring": "OFFICIAL_RUNNER_REQUIRED",
            "preparation_only_not_execution": True,
        }
        for identifier in task_ids
    ]


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
