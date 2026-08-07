#!/usr/bin/env python3
"""Verify a supplemental legacy-submission archive index without writing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

from plan_legacy_submission_isolation import (
    PlanError,
    SHA256_RE,
    _compare_inventory,
    _inventory_tree,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "rc11-supplemental-isolation/v1"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _index_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "index_sha256"}


def index_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        _index_core(payload), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inventory_fingerprint(inventory: dict[str, Any]) -> str:
    core = {
        "directories": inventory.get("directories"),
        "files": inventory.get("files"),
    }
    encoded = json.dumps(
        core, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inventory_with_fingerprint(root: Path) -> dict[str, Any]:
    inventory = _inventory_tree(root)
    return {**inventory, "tree_sha256": _inventory_fingerprint(inventory)}


def _validate_inventory(inventory: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(inventory, dict):
        raise PlanError(f"{label}: inventory must be an object")
    directories = inventory.get("directories")
    files = inventory.get("files")
    if not isinstance(directories, list) or not isinstance(files, list):
        raise PlanError(f"{label}: inventory lists are missing")
    if directories != sorted(directories) or len(directories) != len(set(directories)):
        raise PlanError(f"{label}: directories must be sorted and unique")

    seen_files: set[str] = set()
    for relative in directories:
        path = Path(relative) if isinstance(relative, str) else Path("..")
        if not relative or path.is_absolute() or ".." in path.parts:
            raise PlanError(f"{label}: unsafe directory path: {relative!r}")
    for record in files:
        if not isinstance(record, dict):
            raise PlanError(f"{label}: file record must be an object")
        relative = record.get("path")
        path = Path(relative) if isinstance(relative, str) else Path("..")
        if not relative or path.is_absolute() or ".." in path.parts:
            raise PlanError(f"{label}: unsafe file path: {relative!r}")
        if relative in seen_files:
            raise PlanError(f"{label}: duplicate file path: {relative}")
        seen_files.add(relative)
        if not isinstance(record.get("size"), int) or record["size"] < 0:
            raise PlanError(f"{label}: invalid size for {relative}")
        if not isinstance(record.get("sha256"), str) or not SHA256_RE.fullmatch(
            record["sha256"]
        ):
            raise PlanError(f"{label}: invalid SHA-256 for {relative}")
    if [record["path"] for record in files] != sorted(seen_files):
        raise PlanError(f"{label}: files must be sorted")

    tree_sha256 = inventory.get("tree_sha256")
    if not isinstance(tree_sha256, str) or not SHA256_RE.fullmatch(tree_sha256):
        raise PlanError(f"{label}: tree_sha256 is missing or invalid")
    if tree_sha256 != _inventory_fingerprint(inventory):
        raise PlanError(f"{label}: tree_sha256 does not match the inventory")
    return inventory


def _resolve_index_path(
    repo_root: Path,
    value: Any,
    *,
    label: str,
    expected_parent: Path,
) -> tuple[str, Path]:
    if not isinstance(value, str):
        raise PlanError(f"{label} must be a string")
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or relative.as_posix() != value
        or ".." in relative.parts
        or relative.parent != expected_parent
    ):
        raise PlanError(f"unsafe {label}: {value!r}")
    unresolved = repo_root / relative
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise PlanError(f"{label} escapes repository root: {value!r}") from exc
    return value, unresolved


def _load_artifact(path: Path, *, label: str) -> dict[str, Any]:
    artifact = path / "run_leaderboard.json"
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"{label}: cannot read {artifact}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlanError(f"{label}: run_leaderboard.json must be an object")
    return payload


def _verify_tree(path: Path, expected: dict[str, Any], *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise PlanError(f"{label}: archive tree is missing or unsafe: {path}")
    actual = inventory_with_fingerprint(path)
    mismatches = _compare_inventory(expected, actual)
    if expected["tree_sha256"] != actual["tree_sha256"]:
        mismatches.append(
            {
                "kind": "tree_sha256_changed",
                "expected": expected["tree_sha256"],
                "actual": actual["tree_sha256"],
            }
        )
    if mismatches:
        raise PlanError(f"{label}: inventory mismatch: {mismatches}")
    return actual


def _validate_common_entry(
    entry: Any,
    *,
    repo_root: Path,
    archive_parent: Path,
    label: str,
) -> tuple[str, Path, str, Path, dict[str, Any]]:
    if not isinstance(entry, dict):
        raise PlanError(f"{label}: entry must be an object")
    source_value, source = _resolve_index_path(
        repo_root,
        entry.get("source_path"),
        label=f"{label} source_path",
        expected_parent=Path("submissions"),
    )
    archive_value, archive = _resolve_index_path(
        repo_root,
        entry.get("archive_path"),
        label=f"{label} archive_path",
        expected_parent=archive_parent,
    )
    if Path(source_value).name != Path(archive_value).name:
        raise PlanError(f"{label}: archive name differs from source")
    if source.is_symlink() or source.exists():
        raise PlanError(f"{label}: source path is still active: {source_value}")

    inventory = _validate_inventory(
        entry.get("source_inventory"), label=f"{label} source_inventory"
    )
    _verify_tree(archive, inventory, label=f"{label} archive")
    artifact = _load_artifact(archive, label=f"{label} archive")
    entry_id = entry.get("entry_id")
    if not isinstance(entry_id, str) or not entry_id:
        raise PlanError(f"{label}: entry_id is required")
    if artifact.get("entry_id") != entry_id:
        raise PlanError(f"{label}: archived artifact entry_id does not match index")
    return source_value, source, archive_value, archive, artifact


def verify_supplemental_index(*, repo_root: Path, index_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read supplemental index {index_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlanError("supplemental index must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PlanError(
            f"unsupported schema version: {payload.get('schema_version')!r}"
        )

    archive_date = payload.get("archive_date")
    if not isinstance(archive_date, str):
        raise PlanError("archive_date is required")
    try:
        parsed_date = date.fromisoformat(archive_date)
    except ValueError as exc:
        raise PlanError("archive_date must be a real YYYY-MM-DD date") from exc
    if parsed_date.isoformat() != archive_date:
        raise PlanError("archive_date must use YYYY-MM-DD format")

    benchmark_commit = payload.get("benchmark_commit")
    if not isinstance(benchmark_commit, str) or not GIT_SHA_RE.fullmatch(
        benchmark_commit
    ):
        raise PlanError("benchmark_commit must be a 40-character lowercase SHA")

    fingerprint = payload.get("index_sha256")
    if not isinstance(fingerprint, str) or not SHA256_RE.fullmatch(fingerprint):
        raise PlanError("index_sha256 is required and must be a SHA-256 digest")
    if fingerprint != index_fingerprint(payload):
        raise PlanError("index_sha256 does not match the supplemental index")

    coexistence = payload.get("coexistence")
    missing_target_id = payload.get("missing_target_id")
    if not isinstance(coexistence, list) or not isinstance(missing_target_id, list):
        raise PlanError("coexistence and missing_target_id must be lists")

    coexistence_sources: list[str] = []
    missing_sources: list[str] = []
    archive_paths: list[str] = []
    coexistence_parent = (
        Path("archive") / "legacy" / "superseded-coexistence" / archive_date
    )
    missing_parent = Path("archive") / "legacy" / "missing-target-id" / archive_date

    for position, entry in enumerate(coexistence):
        label = f"coexistence[{position}]"
        source_value, _, archive_value, _, archived_artifact = _validate_common_entry(
            entry,
            repo_root=repo_root,
            archive_parent=coexistence_parent,
            label=label,
        )
        if entry.get("reason") != "superseded-coexistence":
            raise PlanError(f"{label}: invalid reason")
        selection_reason = entry.get("selection_reason")
        if not isinstance(selection_reason, str) or not selection_reason.strip():
            raise PlanError(f"{label}: selection_reason is required")
        retained_value, retained = _resolve_index_path(
            repo_root,
            entry.get("retained_path"),
            label=f"{label} retained_path",
            expected_parent=Path("submissions"),
        )
        retained_inventory = _validate_inventory(
            entry.get("retained_inventory"), label=f"{label} retained_inventory"
        )
        _verify_tree(retained, retained_inventory, label=f"{label} retained")
        retained_artifact = _load_artifact(retained, label=f"{label} retained")
        if retained_artifact.get("entry_id") != archived_artifact.get("entry_id"):
            raise PlanError(f"{label}: retained and archived entry_id differ")
        if retained_value == source_value:
            raise PlanError(f"{label}: retained_path must differ from source_path")
        coexistence_sources.append(source_value)
        archive_paths.append(archive_value)

    for position, entry in enumerate(missing_target_id):
        label = f"missing_target_id[{position}]"
        source_value, _, archive_value, _, artifact = _validate_common_entry(
            entry,
            repo_root=repo_root,
            archive_parent=missing_parent,
            label=label,
        )
        if entry.get("reason") != "metadata.target_id missing":
            raise PlanError(f"{label}: invalid reason")
        errors = entry.get("errors")
        if (
            not isinstance(errors, list)
            or not errors
            or not all(isinstance(error, str) and error for error in errors)
        ):
            raise PlanError(f"{label}: errors must be a non-empty string list")
        if not isinstance(entry.get("spec_id"), str) or not entry["spec_id"]:
            raise PlanError(f"{label}: spec_id is required")
        metadata = artifact.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if metadata.get("target_id"):
            raise PlanError(f"{label}: archived artifact has metadata.target_id")
        missing_sources.append(source_value)
        archive_paths.append(archive_value)

    if coexistence_sources != sorted(coexistence_sources):
        raise PlanError("coexistence source paths must be sorted")
    if missing_sources != sorted(missing_sources):
        raise PlanError("missing_target_id source paths must be sorted")
    source_paths = [*coexistence_sources, *missing_sources]
    if len(source_paths) != len(set(source_paths)):
        raise PlanError("supplemental source paths must be unique")
    if len(archive_paths) != len(set(archive_paths)):
        raise PlanError("supplemental archive paths must be unique")

    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "benchmark_commit": benchmark_commit,
        "archive_date": archive_date,
        "coexistence_count": len(coexistence),
        "missing_target_id_count": len(missing_target_id),
        "entry_count": len(source_paths),
        "index_sha256": fingerprint,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_supplemental_index(
            repo_root=args.repo_root, index_path=args.index
        )
    except PlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
