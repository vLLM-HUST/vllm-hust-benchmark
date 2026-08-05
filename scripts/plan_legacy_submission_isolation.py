#!/usr/bin/env python3
"""Build or verify a read-only legacy-submission isolation plan.

The command never moves, removes, or rewrites submission data. Its JSON output
can be reviewed and redirected to ``index.json`` by a repository writer after
the legacy-isolation decision is approved.

Examples::

    python scripts/plan_legacy_submission_isolation.py \
      --archive-date 2026-08-05
    python scripts/plan_legacy_submission_isolation.py \
      --archive-date 2026-08-05 --verify-index /path/to/index.json

Exit code 0 means the plan and current trees agree. Exit code 2 means a source
hash changed, an archive target conflicts, or the index is invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = "legacy-submission-isolation-index/v1"
SUPPORTED_FAILURE_REASONS = frozenset({"PROVENANCE_INCOMPLETE", "CHECKSUM_INCOMPLETE"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PlanError(ValueError):
    """Raised when an isolation index is unsafe or structurally invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_tree(root: Path) -> dict[str, Any]:
    """Return a deterministic inventory without following symbolic links."""
    if not root.is_dir() or root.is_symlink():
        raise PlanError(f"submission tree is not a real directory: {root}")

    directories: list[str] = []
    files: list[dict[str, Any]] = []
    for current, dir_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(dir_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                raise PlanError(f"symbolic links are not allowed: {path}")
            directories.append(relative)
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink() or not path.is_file():
                raise PlanError(f"non-regular files are not allowed: {path}")
            files.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return {
        "directories": sorted(directories),
        "files": sorted(files, key=lambda item: item["path"]),
    }


def _repository_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise PlanError(f"path escapes repository root: {path}") from exc


def _resolve_relative_root(repo_root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PlanError(f"{label} must be a repository-relative path: {value!r}")
    unresolved = repo_root / candidate
    if unresolved.is_symlink():
        raise PlanError(f"{label} must not be a symbolic link: {value!r}")
    resolved = unresolved.resolve()
    _repository_relative(repo_root, resolved)
    return resolved


def _load_artifact_metadata(submission_dir: Path) -> dict[str, Any]:
    artifact_path = submission_dir / "run_leaderboard.json"
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(
            f"cannot read artifact metadata: {artifact_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PlanError(f"artifact must be a JSON object: {artifact_path}")

    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    runtime = metadata.get("runtime_provenance")
    runtime = runtime if isinstance(runtime, dict) else {}
    engine = runtime.get("engine")
    engine = engine if isinstance(engine, dict) else {}
    plugin = runtime.get("plugin")
    plugin = plugin if isinstance(plugin, dict) else {}
    return {
        "entry_id": payload.get("entry_id"),
        "data_source": metadata.get("data_source"),
        "engine_commit": engine.get("commit") or metadata.get("git_commit"),
        "plugin_commit": plugin.get("commit"),
    }


def _plan_fingerprint(core: dict[str, Any]) -> str:
    encoded = json.dumps(
        core, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "archive_date": payload.get("archive_date"),
        "source_root": payload.get("source_root"),
        "archive_root": payload.get("archive_root"),
        "entries": payload.get("entries"),
    }


def _validate_inventory(inventory: Any, *, entry_name: str) -> None:
    if not isinstance(inventory, dict):
        raise PlanError(f"{entry_name}: inventory must be an object")
    directories = inventory.get("directories")
    files = inventory.get("files")
    if not isinstance(directories, list) or not isinstance(files, list):
        raise PlanError(f"{entry_name}: inventory lists are missing")
    if directories != sorted(directories) or len(directories) != len(set(directories)):
        raise PlanError(f"{entry_name}: directories must be sorted and unique")
    seen_files: set[str] = set()
    for relative in directories:
        path = Path(relative) if isinstance(relative, str) else Path("..")
        if not relative or path.is_absolute() or ".." in path.parts:
            raise PlanError(f"{entry_name}: unsafe directory path: {relative!r}")
    for record in files:
        if not isinstance(record, dict):
            raise PlanError(f"{entry_name}: file record must be an object")
        relative = record.get("path")
        path = Path(relative) if isinstance(relative, str) else Path("..")
        if not relative or path.is_absolute() or ".." in path.parts:
            raise PlanError(f"{entry_name}: unsafe file path: {relative!r}")
        if relative in seen_files:
            raise PlanError(f"{entry_name}: duplicate file path: {relative}")
        seen_files.add(relative)
        if not isinstance(record.get("size"), int) or record["size"] < 0:
            raise PlanError(f"{entry_name}: invalid size for {relative}")
        if not isinstance(record.get("sha256"), str) or not SHA256_RE.match(
            record["sha256"]
        ):
            raise PlanError(f"{entry_name}: invalid SHA-256 for {relative}")
    if [record["path"] for record in files] != sorted(seen_files):
        raise PlanError(f"{entry_name}: files must be sorted")


def _validate_plan(
    payload: Any,
    *,
    source_root_rel: str,
    archive_root_rel: str,
    archive_date: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PlanError("index must be a JSON object")
    core = _plan_core(payload)
    if core["schema_version"] != SCHEMA_VERSION:
        raise PlanError(f"unsupported schema version: {core['schema_version']!r}")
    if core["archive_date"] != archive_date:
        raise PlanError("index archive_date does not match the requested date")
    if core["source_root"] != source_root_rel:
        raise PlanError("index source_root does not match the requested source")
    if core["archive_root"] != archive_root_rel:
        raise PlanError("index archive_root does not match the requested archive")
    if not isinstance(core["entries"], list):
        raise PlanError("index entries must be a list")

    expected_names: list[str] = []
    for entry in core["entries"]:
        if not isinstance(entry, dict):
            raise PlanError("index entry must be an object")
        original = entry.get("original_path")
        archive = entry.get("archive_path")
        if not isinstance(original, str) or not isinstance(archive, str):
            raise PlanError("index entry paths must be strings")
        original_path = Path(original)
        archive_path = Path(archive)
        if original_path.parent.as_posix() != source_root_rel:
            raise PlanError(f"unsafe original path: {original!r}")
        expected_archive_parent = Path(archive_root_rel) / archive_date
        if archive_path.parent != expected_archive_parent:
            raise PlanError(f"unsafe archive path: {archive!r}")
        if archive_path.name != original_path.name:
            raise PlanError(f"archive name differs from source: {archive!r}")
        if entry.get("failure_reason") not in SUPPORTED_FAILURE_REASONS:
            raise PlanError(f"unsupported failure reason in index: {original!r}")
        _validate_inventory(entry.get("inventory"), entry_name=original_path.name)
        expected_names.append(original_path.name)

    if expected_names != sorted(expected_names) or len(expected_names) != len(
        set(expected_names)
    ):
        raise PlanError("index entries must be sorted and unique")
    fingerprint = payload.get("plan_sha256")
    if not isinstance(fingerprint, str) or not SHA256_RE.match(fingerprint):
        raise PlanError("index plan_sha256 is required and must be a SHA-256 digest")
    if fingerprint != _plan_fingerprint(core):
        raise PlanError("index plan_sha256 does not match its immutable fields")
    return core


def _compare_inventory(
    expected: dict[str, Any], actual: dict[str, Any]
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    expected_dirs = set(expected["directories"])
    actual_dirs = set(actual["directories"])
    for relative in sorted(expected_dirs - actual_dirs):
        mismatches.append({"kind": "missing_directory", "path": relative})
    for relative in sorted(actual_dirs - expected_dirs):
        mismatches.append({"kind": "unexpected_directory", "path": relative})

    expected_files = {item["path"]: item for item in expected["files"]}
    actual_files = {item["path"]: item for item in actual["files"]}
    for relative in sorted(expected_files.keys() - actual_files.keys()):
        mismatches.append({"kind": "missing_file", "path": relative})
    for relative in sorted(actual_files.keys() - expected_files.keys()):
        mismatches.append({"kind": "unexpected_file", "path": relative})
    for relative in sorted(expected_files.keys() & actual_files.keys()):
        expected_file = expected_files[relative]
        actual_file = actual_files[relative]
        if expected_file["size"] != actual_file["size"]:
            mismatches.append(
                {
                    "kind": "size_changed",
                    "path": relative,
                    "expected": expected_file["size"],
                    "actual": actual_file["size"],
                }
            )
        if expected_file["sha256"] != actual_file["sha256"]:
            mismatches.append(
                {
                    "kind": "sha256_changed",
                    "path": relative,
                    "expected": expected_file["sha256"],
                    "actual": actual_file["sha256"],
                }
            )
    return mismatches


def _scan_failures(source_root: Path) -> list[dict[str, Any]]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from vllm_hust_benchmark.integration import (  # noqa: PLC0415
        _scan_submission_admission_failures,
    )

    return _scan_submission_admission_failures(source_root)


def _verify_core(
    core: dict[str, Any],
    *,
    repo_root: Path,
    source_root: Path,
    archive_date_root: Path,
) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    states: list[dict[str, str]] = []
    planned_names = {Path(entry["original_path"]).name for entry in core["entries"]}

    current_failures = _scan_failures(source_root)
    current_by_name = {Path(item["dir"]).name: item for item in current_failures}
    for name in sorted(current_by_name.keys() - planned_names):
        errors.append(
            {
                "kind": "unindexed_admission_failure",
                "path": _repository_relative(repo_root, source_root / name),
                "reason": current_by_name[name]["reason"],
            }
        )

    for entry in core["entries"]:
        name = Path(entry["original_path"]).name
        source = source_root / name
        target = archive_date_root / name
        source_exists = source.exists() or source.is_symlink()
        target_exists = target.exists() or target.is_symlink()
        source_matches = False
        target_matches = False

        if source_exists:
            try:
                source_diff = _compare_inventory(
                    entry["inventory"], _inventory_tree(source)
                )
            except PlanError as exc:
                source_diff = [{"kind": "unsafe_tree", "detail": str(exc)}]
            source_matches = not source_diff
            if source_diff:
                errors.append(
                    {
                        "kind": "source_hash_changed",
                        "path": entry["original_path"],
                        "mismatches": source_diff,
                    }
                )
            current = current_by_name.get(name)
            if current is None:
                errors.append(
                    {
                        "kind": "source_no_longer_fails_admission",
                        "path": entry["original_path"],
                    }
                )
            elif current["reason"] != entry["failure_reason"]:
                errors.append(
                    {
                        "kind": "admission_reason_changed",
                        "path": entry["original_path"],
                        "expected": entry["failure_reason"],
                        "actual": current["reason"],
                    }
                )

        if target_exists:
            try:
                target_diff = _compare_inventory(
                    entry["inventory"], _inventory_tree(target)
                )
            except PlanError as exc:
                target_diff = [{"kind": "unsafe_tree", "detail": str(exc)}]
            target_matches = not target_diff
            if target_diff:
                errors.append(
                    {
                        "kind": "archive_target_conflict",
                        "path": entry["archive_path"],
                        "mismatches": target_diff,
                    }
                )

        if source_exists and source_matches and not target_exists:
            state = "planned"
        elif source_exists and source_matches and target_exists and target_matches:
            state = "copied_source_still_active"
        elif not source_exists and target_exists and target_matches:
            state = "already_archived"
        elif not source_exists and not target_exists:
            state = "missing_source_and_target"
            errors.append(
                {
                    "kind": state,
                    "path": entry["original_path"],
                }
            )
        else:
            state = "conflict"
        states.append({"path": entry["original_path"], "state": state})

    return {
        "ok": not errors,
        "entry_count": len(core["entries"]),
        "states": states,
        "errors": errors,
    }


def build_or_verify_plan(
    *,
    repo_root: Path,
    source_root_value: str,
    archive_root_value: str,
    archive_date: str,
    existing_index: Path | None = None,
) -> dict[str, Any]:
    """Build a new plan or verify an existing immutable index."""
    try:
        parsed_date = date.fromisoformat(archive_date)
    except ValueError as exc:
        raise PlanError(
            "archive_date must be a real date in YYYY-MM-DD format"
        ) from exc
    if parsed_date.isoformat() != archive_date:
        raise PlanError("archive_date must use YYYY-MM-DD format")
    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise PlanError(f"repository root is not a directory: {repo_root}")
    source_root = _resolve_relative_root(
        repo_root, source_root_value, label="source root"
    )
    archive_root = _resolve_relative_root(
        repo_root, archive_root_value, label="archive root"
    )
    source_root_rel = _repository_relative(repo_root, source_root)
    archive_root_rel = _repository_relative(repo_root, archive_root)
    if not source_root.is_dir():
        raise PlanError(f"source root is not a directory: {source_root_rel}")
    if source_root == archive_root or source_root in archive_root.parents:
        raise PlanError("archive root must not contain the active source root")
    if archive_root == source_root or archive_root in source_root.parents:
        raise PlanError("archive root must not be inside the active source root")
    archive_date_root = archive_root / archive_date
    if archive_date_root.is_symlink():
        raise PlanError("dated archive root must not be a symbolic link")
    _repository_relative(repo_root, archive_date_root)
    automatic_index = archive_date_root / "index.json"
    if existing_index is None and automatic_index.is_file():
        existing_index = automatic_index

    if existing_index is not None:
        try:
            payload = json.loads(existing_index.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanError(
                f"cannot read existing index: {existing_index}: {exc}"
            ) from exc
        core = _validate_plan(
            payload,
            source_root_rel=source_root_rel,
            archive_root_rel=archive_root_rel,
            archive_date=archive_date,
        )
    else:
        failures = _scan_failures(source_root)
        unsupported = [
            item for item in failures if item["reason"] not in SUPPORTED_FAILURE_REASONS
        ]
        if unsupported:
            reasons = ", ".join(
                f"{Path(item['dir']).name}:{item['reason']}" for item in unsupported
            )
            raise PlanError(f"unsupported admission failures require triage: {reasons}")

        entries: list[dict[str, Any]] = []
        for failure in failures:
            source = Path(failure["dir"])
            metadata = _load_artifact_metadata(source)
            entries.append(
                {
                    "original_path": _repository_relative(repo_root, source),
                    "archive_path": _repository_relative(
                        repo_root, archive_date_root / source.name
                    ),
                    "failure_reason": failure["reason"],
                    "failure_detail": failure["detail"],
                    **metadata,
                    "inventory": _inventory_tree(source),
                }
            )
        entries.sort(key=lambda item: item["original_path"])
        core = {
            "schema_version": SCHEMA_VERSION,
            "archive_date": archive_date,
            "source_root": source_root_rel,
            "archive_root": archive_root_rel,
            "entries": entries,
        }

    result = {
        **core,
        "plan_sha256": _plan_fingerprint(core),
        "verification": _verify_core(
            core,
            repo_root=repo_root,
            source_root=source_root,
            archive_date_root=archive_date_root,
        ),
    }

    if archive_date_root.is_dir():
        allowed = {Path(entry["archive_path"]).name for entry in core["entries"]}
        allowed.update({"index.json", "README.md"})
        unexpected = sorted(
            child.name
            for child in archive_date_root.iterdir()
            if child.name not in allowed
        )
        for name in unexpected:
            result["verification"]["errors"].append(
                {
                    "kind": "unindexed_archive_target",
                    "path": _repository_relative(repo_root, archive_date_root / name),
                }
            )
        result["verification"]["ok"] = not result["verification"]["errors"]
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-date", required=True, help="Archive date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--repo-root", default=str(REPO_ROOT), help="Benchmark repository root."
    )
    parser.add_argument("--source-root", default="submissions")
    parser.add_argument("--archive-root", default="archive/legacy/incomplete-evidence")
    parser.add_argument(
        "--verify-index",
        type=Path,
        help="Verify against an existing generated index instead of creating a new plan.",
    )
    args = parser.parse_args(argv)

    try:
        result = build_or_verify_plan(
            repo_root=Path(args.repo_root),
            source_root_value=args.source_root,
            archive_root_value=args.archive_root,
            archive_date=args.archive_date,
            existing_index=args.verify_index,
        )
    except PlanError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True))
    return 0 if result["verification"]["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
