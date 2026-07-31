#!/usr/bin/env python3
"""Generate an admission report for leaderboard snapshot entries.

For each entry in the snapshot, the script looks up a matching fixed-target
profile and produces an admission decision (keep / quarantine / specialty).
The report is written as a JSON file matching ``admission-report/v1``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_hust_benchmark.fixed_target_registry import (  # noqa: E402
    FixedTargetProfile,
    find_matching_profile,
    load_fixed_target_registry,
)


SCHEMA_VERSION = "admission-report/v1"
DISPOSITION_KEEP = "keep"
DISPOSITION_QUARANTINE = "quarantine"
DISPOSITION_SPECIALTY = "specialty"
DISPOSITION_RERUN = "rerun"
DISPOSITION_ORDER = (
    DISPOSITION_KEEP,
    DISPOSITION_QUARANTINE,
    DISPOSITION_SPECIALTY,
    DISPOSITION_RERUN,
)

# Effective server parameters that active profiles pin down.
_ACTIVE_CHECKED_FIELDS: tuple[tuple[str, Any], ...] = (
    ("gpu_memory_utilization", None),
    ("max_model_len", None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an admission report for leaderboard snapshot entries."
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Leaderboard snapshot JSON file path (e.g. leaderboard-data/snapshots/leaderboard_single.json).",
    )
    parser.add_argument(
        "--registry",
        required=True,
        help="Fixed-target registry JSON file path.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output admission report JSON file path.",
    )
    return parser.parse_args()


def load_snapshot(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"{path} must contain a JSON array")
    return [entry for entry in payload if isinstance(entry, dict)]


def resolve_registry_version(registry: tuple[FixedTargetProfile, ...]) -> str:
    """Derive a registry version string from the loaded profiles.

    All profiles in a campaign share the same ``target_id``; we join the
    unique values so the report records which campaign(s) were applied.
    """
    target_ids: list[str] = []
    seen: set[str] = set()
    for profile in registry:
        if profile.target_id and profile.target_id not in seen:
            target_ids.append(profile.target_id)
            seen.add(profile.target_id)
    if target_ids:
        return ",".join(target_ids)
    return "unknown"


def _resolve_entry_artifact_path(entry: Mapping[str, Any]) -> str | None:
    """Resolve the original submission artifact path for a snapshot entry."""
    metadata = entry.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    for key in ("artifact_path", "submission_dir"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _extract_scenario(entry: Mapping[str, Any]) -> str:
    workload = (
        entry.get("workload") if isinstance(entry.get("workload"), Mapping) else {}
    )
    name = workload.get("name")
    if name:
        return str(name)
    same_spec = (
        entry.get("same_spec") if isinstance(entry.get("same_spec"), Mapping) else {}
    )
    scenario = same_spec.get("scenario")
    if scenario:
        return str(scenario)
    return ""


def _extract_actual_config(entry: Mapping[str, Any]) -> dict[str, Any]:
    model = entry.get("model") if isinstance(entry.get("model"), Mapping) else {}
    hardware = (
        entry.get("hardware") if isinstance(entry.get("hardware"), Mapping) else {}
    )
    workload = (
        entry.get("workload") if isinstance(entry.get("workload"), Mapping) else {}
    )
    same_spec = (
        entry.get("same_spec") if isinstance(entry.get("same_spec"), Mapping) else {}
    )
    server = same_spec.get("resolved_server_parameters")
    server = server if isinstance(server, Mapping) else {}
    return {
        "model": model.get("repo_id"),
        "hardware_chip_model": hardware.get("chip_model"),
        "chip_count": hardware.get("chip_count"),
        "model_precision": model.get("precision"),
        "gpu_memory_utilization": server.get("gpu_memory_utilization"),
        "max_model_len": server.get("max_model_len"),
        "workload_name": workload.get("name"),
    }


def _build_required_config(profile: FixedTargetProfile) -> dict[str, Any]:
    return {
        "model": profile.model,
        "hardware_chip_model": profile.hardware_chip_model,
        "chip_count": profile.chip_count,
        "model_precision": profile.model_precision,
        "gpu_memory_utilization": profile.gpu_memory_utilization,
        "max_model_len": profile.max_model_len,
        "workload_name": profile.workload_name,
    }


def _fixed_target_numeric_equal(left: Any, right: Any) -> bool:
    """Compare two values numerically, falling back to direct equality."""
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return left == right


def _build_report_entry(
    entry: Mapping[str, Any],
    registry: tuple[FixedTargetProfile, ...],
) -> dict[str, Any]:
    entry_id = str(entry.get("entry_id") or "")
    scenario = _extract_scenario(entry)
    actual_config = _extract_actual_config(entry)
    artifact_path = _resolve_entry_artifact_path(entry)

    profile = find_matching_profile(entry, registry)

    base_record = {
        "entry_id": entry_id,
        "scenario": scenario,
        "actual_config": actual_config,
        "artifact_path": artifact_path,
    }

    if profile is None:
        return {
            **base_record,
            "profile_name": None,
            "required_config": None,
            "missing_fields": [],
            "drift_fields": [],
            "disposition": DISPOSITION_KEEP,
            "reason": "no matching fixed-target profile (non-official entry)",
        }

    if profile.status == "specialty":
        return {
            **base_record,
            "profile_name": profile.profile_name,
            "required_config": None,
            "missing_fields": [],
            "drift_fields": [],
            "disposition": DISPOSITION_SPECIALTY,
            "reason": f"specialty target {profile.profile_name!r} has no fixed contract",
        }

    if profile.status == "retired":
        return {
            **base_record,
            "profile_name": profile.profile_name,
            "required_config": None,
            "missing_fields": [],
            "drift_fields": [],
            "disposition": DISPOSITION_QUARANTINE,
            "reason": f"retired target {profile.profile_name!r}",
        }

    # status == "active"
    required_config = _build_required_config(profile)
    same_spec = (
        entry.get("same_spec") if isinstance(entry.get("same_spec"), Mapping) else {}
    )
    server = same_spec.get("resolved_server_parameters")
    server = server if isinstance(server, Mapping) else {}

    missing_fields: list[str] = []
    drift_fields: list[str] = []
    for field_name, _placeholder in _ACTIVE_CHECKED_FIELDS:
        required_value = getattr(profile, field_name)
        if field_name not in server:
            missing_fields.append(field_name)
            continue
        actual_value = server[field_name]
        if not _fixed_target_numeric_equal(actual_value, required_value):
            drift_fields.append(field_name)

    if missing_fields:
        return {
            **base_record,
            "profile_name": profile.profile_name,
            "required_config": required_config,
            "missing_fields": missing_fields,
            "drift_fields": drift_fields,
            "disposition": DISPOSITION_QUARANTINE,
            "reason": f"missing required field(s): {', '.join(missing_fields)}",
        }

    if drift_fields:
        return {
            **base_record,
            "profile_name": profile.profile_name,
            "required_config": required_config,
            "missing_fields": missing_fields,
            "drift_fields": drift_fields,
            "disposition": DISPOSITION_QUARANTINE,
            "reason": f"config drift on field(s): {', '.join(drift_fields)}",
        }

    return {
        **base_record,
        "profile_name": profile.profile_name,
        "required_config": required_config,
        "missing_fields": [],
        "drift_fields": [],
        "disposition": DISPOSITION_KEEP,
        "reason": None,
    }


def print_summary(entries: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {disposition: 0 for disposition in DISPOSITION_ORDER}
    for entry in entries:
        disposition = entry.get("disposition", DISPOSITION_KEEP)
        counts[disposition] = counts.get(disposition, 0) + 1

    print("admission report summary:", file=sys.stderr)
    for disposition in DISPOSITION_ORDER:
        print(f"  {disposition}: {counts.get(disposition, 0)}", file=sys.stderr)
    print(f"total: {len(entries)}", file=sys.stderr)


def main() -> int:
    args = parse_args()

    snapshot_path = Path(args.snapshot)
    registry_path = Path(args.registry)
    output_path = Path(args.output)

    if not snapshot_path.is_file():
        print(f"error: missing snapshot file: {snapshot_path}", file=sys.stderr)
        return 1
    if not registry_path.is_file():
        print(f"error: missing registry file: {registry_path}", file=sys.stderr)
        return 1

    try:
        registry = load_fixed_target_registry(registry_path)
    except ValueError as exc:
        print(f"error: failed to load registry: {exc}", file=sys.stderr)
        return 1

    registry_version = resolve_registry_version(registry)
    entries = load_snapshot(snapshot_path)

    report_entries = [_build_report_entry(entry, registry) for entry in entries]

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "registry_version": registry_version,
        "entries": report_entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print_summary(report_entries)
    print(f"wrote admission report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
