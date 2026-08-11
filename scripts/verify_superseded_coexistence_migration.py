#!/usr/bin/env python3
"""Validate a superseded-coexistence migration index without writing data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "superseded-coexistence-migration/v1"
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
NUMERIC_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
EFFECTIVE_SPEC_KEYS = (
    "schema_version",
    "spec_id",
    "scenario",
    "model",
    "model_parameters",
    "model_precision",
    "model_quantization",
    "hardware_vendor",
    "hardware_chip_model",
    "chip_count",
    "node_count",
    "resolved_server_parameters",
    "resolved_client_parameters",
)


class MigrationIndexError(ValueError):
    """Raised when migration audit metadata does not match repository state."""


def _normalize_parameter_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_parameter_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_parameter_value(item) for item in value]
    if isinstance(value, str) and NUMERIC_RE.fullmatch(value):
        return float(value) if "." in value else int(value)
    return value


def effective_spec_fingerprint(payload: dict[str, Any]) -> str:
    """Hash effective identity while normalizing numeric string representations."""
    missing = [key for key in EFFECTIVE_SPEC_KEYS if key not in payload]
    if missing:
        raise MigrationIndexError(
            f"effective same-spec is missing required fields: {', '.join(missing)}"
        )
    for key in ("resolved_server_parameters", "resolved_client_parameters"):
        if not isinstance(payload[key], dict):
            raise MigrationIndexError(f"effective same-spec {key} must be an object")
    normalized = {
        key: _normalize_parameter_value(deepcopy(payload[key]))
        for key in EFFECTIVE_SPEC_KEYS
    }
    encoded = json.dumps(
        normalized, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationIndexError(f"{label}: cannot read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MigrationIndexError(f"{label}: JSON payload must be an object")
    return payload


def _safe_directory(
    repo_root: Path,
    value: Any,
    *,
    label: str,
    expected_parent: Path,
) -> tuple[str, Path]:
    if not isinstance(value, str):
        raise MigrationIndexError(f"{label} must be a string")
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or relative.as_posix() != value
        or ".." in relative.parts
        or relative.parent != expected_parent
    ):
        raise MigrationIndexError(f"unsafe {label}: {value!r}")

    path = repo_root / relative
    resolved = path.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise MigrationIndexError(
            f"{label} escapes repository root: {value!r}"
        ) from exc

    current = repo_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise MigrationIndexError(f"{label} contains a symlink: {value!r}")
    if not path.is_dir():
        raise MigrationIndexError(f"{label} is not a directory: {value!r}")
    return value, path


def _supersedes_set(metadata: dict[str, Any]) -> set[str]:
    value = metadata.get("supersedes")
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {str(item) for item in value if item}
    return set()


def verify_migration_index(*, repo_root: Path, index_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    index_path = index_path if index_path.is_absolute() else repo_root / index_path
    if index_path.is_symlink():
        raise MigrationIndexError("migration index must not be a symlink")
    payload = _load_json(index_path, label="migration index")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MigrationIndexError(
            f"unsupported schema version: {payload.get('schema_version')!r}"
        )

    archive_date = payload.get("archive_date")
    if not isinstance(archive_date, str):
        raise MigrationIndexError("archive_date is required")
    try:
        parsed_date = date.fromisoformat(archive_date)
    except ValueError as exc:
        raise MigrationIndexError(
            "archive_date must be a real YYYY-MM-DD date"
        ) from exc
    if parsed_date.isoformat() != archive_date:
        raise MigrationIndexError("archive_date must use YYYY-MM-DD format")

    benchmark_commit = payload.get("benchmark_commit")
    if not isinstance(benchmark_commit, str) or not GIT_SHA_RE.fullmatch(
        benchmark_commit
    ):
        raise MigrationIndexError(
            "benchmark_commit must be a 40-character lowercase SHA"
        )
    if payload.get("production_fallback") is not False:
        raise MigrationIndexError("production_fallback must be false")

    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise MigrationIndexError("entries must be a non-empty list")

    archive_parent = (
        Path("archive")
        / "suspect"
        / f"superseded-coexistence-historical-pr-backfill-{archive_date.replace('-', '')}"
    )
    expected_index = repo_root / archive_parent / "index.json"
    if index_path.resolve() != expected_index.resolve():
        raise MigrationIndexError(f"index path must be {expected_index}")

    archived_ids: list[str] = []
    retained_ids: list[str] = []
    archived_paths: list[str] = []
    retained_paths: list[str] = []
    for position, entry in enumerate(entries):
        label = f"entries[{position}]"
        if not isinstance(entry, dict):
            raise MigrationIndexError(f"{label} must be an object")
        archived_id = entry.get("archived_entry_id")
        retained_id = entry.get("retained_entry_id")
        if not isinstance(archived_id, str) or not archived_id:
            raise MigrationIndexError(f"{label}.archived_entry_id is required")
        if not isinstance(retained_id, str) or not retained_id:
            raise MigrationIndexError(f"{label}.retained_entry_id is required")
        if archived_id == retained_id:
            raise MigrationIndexError(f"{label} must reference two different entries")

        archived_value, archived = _safe_directory(
            repo_root,
            entry.get("archived_path"),
            label=f"{label}.archived_path",
            expected_parent=archive_parent,
        )
        retained_value, retained = _safe_directory(
            repo_root,
            entry.get("retained_path"),
            label=f"{label}.retained_path",
            expected_parent=Path("submissions"),
        )
        source = repo_root / "submissions" / archived.name
        if source.is_symlink() or source.exists():
            raise MigrationIndexError(
                f"{label}: archived source is still active: submissions/{archived.name}"
            )

        archived_artifact = _load_json(
            archived / "run_leaderboard.json", label=f"{label} archived artifact"
        )
        retained_artifact = _load_json(
            retained / "run_leaderboard.json", label=f"{label} retained artifact"
        )
        if archived_artifact.get("entry_id") != archived_id:
            raise MigrationIndexError(
                f"{label}: archived entry_id does not match index"
            )
        if retained_artifact.get("entry_id") != retained_id:
            raise MigrationIndexError(
                f"{label}: retained entry_id does not match index"
            )
        metadata = retained_artifact.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if archived_id not in _supersedes_set(metadata):
            raise MigrationIndexError(
                f"{label}: retained metadata.supersedes does not reference archived entry"
            )
        reason = entry.get("selection_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise MigrationIndexError(f"{label}.selection_reason is required")

        expected_fingerprint = entry.get("effective_spec_sha256")
        if not isinstance(expected_fingerprint, str) or not SHA256_RE.fullmatch(
            expected_fingerprint
        ):
            raise MigrationIndexError(f"{label}.effective_spec_sha256 is invalid")
        archived_spec = _load_json(
            archived / "resolved_same_spec.json", label=f"{label} archived same-spec"
        )
        retained_spec = _load_json(
            retained / "resolved_same_spec.json", label=f"{label} retained same-spec"
        )
        archived_fingerprint = effective_spec_fingerprint(archived_spec)
        retained_fingerprint = effective_spec_fingerprint(retained_spec)
        if archived_fingerprint != retained_fingerprint:
            raise MigrationIndexError(
                f"{label}: archived and retained effective specs do not match"
            )
        if expected_fingerprint != archived_fingerprint:
            raise MigrationIndexError(
                f"{label}: effective_spec_sha256 does not match repository data"
            )

        archived_ids.append(archived_id)
        retained_ids.append(retained_id)
        archived_paths.append(archived_value)
        retained_paths.append(retained_value)

    if archived_ids != sorted(archived_ids):
        raise MigrationIndexError("entries must be sorted by archived_entry_id")
    for label, values in (
        ("archived entry IDs", archived_ids),
        ("retained entry IDs", retained_ids),
        ("archived paths", archived_paths),
        ("retained paths", retained_paths),
    ):
        if len(values) != len(set(values)):
            raise MigrationIndexError(f"{label} must be unique")

    return {
        "ok": True,
        "schema_version": SCHEMA_VERSION,
        "benchmark_commit": benchmark_commit,
        "archive_date": archive_date,
        "entry_count": len(entries),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_migration_index(repo_root=args.repo_root, index_path=args.index)
    except MigrationIndexError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
