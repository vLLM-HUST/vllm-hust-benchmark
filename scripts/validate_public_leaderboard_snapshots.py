#!/usr/bin/env python3
"""Validate that public leaderboard snapshots do not include retired baselines."""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_hust_benchmark.official_targets import build_registry  # noqa: E402
from vllm_hust_benchmark.same_spec import compute_resolved_spec_hash  # noqa: E402
from vllm_hust_benchmark.workload_config_contract import (  # noqa: E402
    WORKLOAD_CONFIG_CONTRACT_VERSION,
    requires_workload_config_contract,
    uses_frozen_prefix_repetition_contract,
    validate_explicit_workload_config,
)

PUBLIC_BASELINE_ENGINE = "vllm"
PUBLIC_BASELINE_VERSION = "0.18.0"
PUBLIC_BASELINE_CHIP = "910B2"
RETIRED_PUBLIC_MODELS = ("Qwen/Qwen3-8B",)
RETIRED_PUBLIC_PRECISIONS = ("BF16",)
RETIRED_BASELINE_TOKENS = ("v0.11.0", "v0110", "0.11.0")
OFFICIAL_PUBLIC_WORKLOADS = {
    "instructcoder-online",
    "prefix-repetition-online",
    "random-latency",
    "random-online",
    "sharegpt-online",
    "sharegpt-throughput",
    "sonnet-throughput",
    "visionarena-online",
}
OFFICIAL_V0180_SPEC_PREFIX = "official-ascend-jan-2026-v0.18.0-"
DEFAULT_PUBLIC_GPU_MEMORY_UTILIZATION = 0.6
RETIRED_PUBLIC_MAX_MODEL_LEN = 30720
SNAPSHOT_FILES = (
    "leaderboard_single.json",
    "leaderboard_multi.json",
)

# Files scanned for suspect-commit exclusion enforcement. Unlike SNAPSHOT_FILES
# (which validates entry lists and requires array payloads), this also includes
# leaderboard_compare.json — a dict payload that is likewise synced to the public
# website and must never surface a suspect commit.
PUBLIC_COMMIT_SCAN_FILES = SNAPSHOT_FILES + ("leaderboard_compare.json",)


@functools.lru_cache(maxsize=1)
def specialty_spec_ids() -> frozenset[str]:
    """Registry-verified specialty spec ids (issue #178).

    Specialty series (e.g. the 910B3 full-graph-parallel dual-stream records)
    are exempt from the official v0.18.0 same-spec pairing check, but only when
    the spec is actually registered with intended_use=specialty via the official
    targets registry — a bare ``specialty-`` string prefix is not trusted
    (PR #172 review round 2: arbitrary unregistered ``specialty-foo`` ids must
    not bypass the gate). Cached: validate_entry is called per entry and the
    registry is identical for the whole run.
    """
    try:
        registry = build_registry(REPO_ROOT)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"WARNING: official targets registry unavailable: {exc}")
        return frozenset()
    return frozenset(
        target["target_id"]
        for target in registry.get("targets", [])
        if target.get("intended_use") == "specialty"
    )


# Issue #79 P0a: explicit entry-level quarantine gate.
# These random-latency records have invalid metrics or mismatched
# client/server configs and must never re-enter public snapshots.
# Original artifacts are retained in archive/suspect/ for audit.
QUARANTINED_ENTRY_IDS = frozenset(
    {
        "83a812e9-5af5-4c2f-8acb-152cc347e0be",
        "13e0c174-976d-4644-94b9-a54573183f3c",
    }
)
REJECTED_SUPERSEDED_REPORT_FILE = "rejected_superseded_report.json"
REJECTED_SUPERSEDED_REPORT_SCHEMA = (
    REPO_ROOT / "schemas" / "rejected_superseded_report_v1.schema.json"
)
REJECTED_SUPERSEDED_REPORT_REQUIRED_FIELDS = (
    "schema_version",
    "generated_at",
    "rejected_submissions",
    "superseded_entries",
    "excluded_plugin_commits",
)
REJECTED_SUPERSEDED_REPORT_SCHEMA_VERSION = "rejected-superseded-report/v1"

QUARANTINE_FILE = "quarantine_leaderboard_entries.json"
QUARANTINE_SUSPECT_SCHEMA = (
    REPO_ROOT / "schemas" / "quarantine_suspect_entries_v2.schema.json"
)
QUARANTINE_SUSPECT_SCHEMA_VERSION = "issue-146-suspect/v2"
QUARANTINE_ADDITIVE_SECTIONS = ("issue_146_suspect_entries",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate curated public leaderboard snapshot files."
    )
    parser.add_argument(
        "--snapshot-dir",
        default="leaderboard-data/snapshots",
        help="Directory containing public leaderboard snapshot JSON files.",
    )
    return parser.parse_args()


def load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"{path} must contain a JSON array")
    return [entry for entry in payload if isinstance(entry, dict)]


def contains_retired_baseline_token(value: Any) -> bool:
    normalized = str(value or "")
    return any(token in normalized for token in RETIRED_BASELINE_TOKENS)


def parse_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate_entry(entry: dict[str, Any], *, source: Path) -> list[str]:
    errors: list[str] = []
    entry_id = str(entry.get("entry_id") or "<missing-entry-id>")
    engine = str(entry.get("engine") or "").strip().lower()
    engine_version = str(entry.get("engine_version") or "").strip()
    workload = str((entry.get("workload") or {}).get("name") or "").strip()
    model = entry.get("model") if isinstance(entry.get("model"), dict) else {}
    hardware = entry.get("hardware") if isinstance(entry.get("hardware"), dict) else {}
    same_spec = (
        entry.get("same_spec") if isinstance(entry.get("same_spec"), dict) else {}
    )
    resolved_server_parameters = (
        same_spec.get("resolved_server_parameters")
        if isinstance(same_spec.get("resolved_server_parameters"), dict)
        else {}
    )
    spec_id = str(same_spec.get("spec_id") or "")
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}

    if entry_id in QUARANTINED_ENTRY_IDS:
        errors.append(
            f"{source.name}:{entry_id}: quarantined entry must not appear in "
            f"public snapshot (issue #79: invalid random-latency record)"
        )

    if engine == PUBLIC_BASELINE_ENGINE and engine_version != PUBLIC_BASELINE_VERSION:
        errors.append(
            f"{source.name}:{entry_id}: public vllm baseline must be "
            f"{PUBLIC_BASELINE_VERSION}, got {engine_version!r}"
        )

    if contains_retired_baseline_token(engine_version):
        errors.append(
            f"{source.name}:{entry_id}: retired baseline version in engine_version "
            f"{engine_version!r}"
        )

    if contains_retired_baseline_token(spec_id):
        errors.append(f"{source.name}:{entry_id}: retired baseline spec_id {spec_id!r}")

    entry_model_name = str(model.get("name") or model.get("repo_id") or "")
    entry_precision_value = str(model.get("precision") or "")
    if entry_model_name in RETIRED_PUBLIC_MODELS:
        errors.append(
            f"{source.name}:{entry_id}: retired public model {entry_model_name!r}"
        )
    if entry_precision_value in RETIRED_PUBLIC_PRECISIONS:
        errors.append(
            f"{source.name}:{entry_id}: retired public precision "
            f"{entry_precision_value!r}"
        )

    if engine == "vllm-hust" and workload in OFFICIAL_PUBLIC_WORKLOADS and not spec_id:
        errors.append(
            f"{source.name}:{entry_id}: public vllm-hust official workload "
            f"{workload!r} must include same_spec"
        )

    if (
        engine == "vllm-hust"
        and workload in OFFICIAL_PUBLIC_WORKLOADS
        and spec_id
        and not spec_id.startswith(OFFICIAL_V0180_SPEC_PREFIX)
        and spec_id not in specialty_spec_ids()
    ):
        errors.append(
            f"{source.name}:{entry_id}: public vllm-hust official workload "
            f"{workload!r} must use official v0.18.0 same_spec, got {spec_id!r}"
        )

    if spec_id.startswith(OFFICIAL_V0180_SPEC_PREFIX):
        gpu_memory_utilization = parse_optional_float(
            resolved_server_parameters.get("gpu_memory_utilization")
        )
        expected_gpu_memory_utilization = (
            0.9
            if uses_frozen_prefix_repetition_contract(entry)
            else DEFAULT_PUBLIC_GPU_MEMORY_UTILIZATION
        )
        if (
            gpu_memory_utilization is not None
            and gpu_memory_utilization != expected_gpu_memory_utilization
        ):
            errors.append(
                f"{source.name}:{entry_id}: official public snapshot must not "
                f"publish non-default gpu_memory_utilization="
                f"{resolved_server_parameters.get('gpu_memory_utilization')!r}; "
                f"rerun with {expected_gpu_memory_utilization} or mark the "
                "record outside the default public snapshot"
            )

        max_model_len = parse_optional_int(
            resolved_server_parameters.get("max_model_len")
        )
        if max_model_len == RETIRED_PUBLIC_MAX_MODEL_LEN:
            errors.append(
                f"{source.name}:{entry_id}: official public snapshot must not "
                f"publish retired max_model_len={RETIRED_PUBLIC_MAX_MODEL_LEN}; "
                "rerun with max_model_len=32768 or mark the record outside the "
                "default public snapshot"
            )

        expected_chip = PUBLIC_BASELINE_CHIP if spec_id.endswith("-910b2") else None
        entry_precision = str(model.get("precision") or "")
        spec_precision = str(same_spec.get("model_precision") or "")
        entry_chip = str(hardware.get("chip_model") or "")
        spec_chip = str(same_spec.get("hardware_chip_model") or "")

        if not spec_precision or entry_precision != spec_precision:
            errors.append(
                f"{source.name}:{entry_id}: official v0.18.0 public spec "
                f"precision must match same_spec; entry={entry_precision!r} "
                f"same_spec={spec_precision!r}"
            )
        if expected_chip and (
            entry_chip != expected_chip or spec_chip != expected_chip
        ):
            errors.append(
                f"{source.name}:{entry_id}: official v0.18.0 public spec "
                f"{spec_id!r} must be {expected_chip}; entry={entry_chip!r} "
                f"same_spec={spec_chip!r}"
            )

    contract_version = str(metadata.get("workload_config_contract") or "")
    historical_unverified = (
        metadata.get("official_admission_status") == "historical-unverified"
    )
    if historical_unverified:
        if metadata.get("verified") is not False:
            errors.append(
                f"{source.name}:{entry_id}: historical-unverified entry must set "
                "metadata.verified=false"
            )
        claimed_fields = [
            field
            for field in (
                "target_id",
                "target_version",
                "profile_id",
                "target_registry_sha256",
            )
            if metadata.get(field)
        ]
        if claimed_fields:
            errors.append(
                f"{source.name}:{entry_id}: historical-unverified entry cannot "
                "claim " + ", ".join(claimed_fields)
            )
        if not str(metadata.get("official_admission_reason") or "").strip():
            errors.append(
                f"{source.name}:{entry_id}: historical-unverified entry requires "
                "metadata.official_admission_reason"
            )

    if requires_workload_config_contract(entry) or contract_version:
        for message in validate_explicit_workload_config(
            entry, validate_target_metadata=not historical_unverified
        ):
            errors.append(
                f"{source.name}:{entry_id}: workload config contract "
                f"{WORKLOAD_CONFIG_CONTRACT_VERSION}: {message}"
            )

    return errors


def validate_rejected_superseded_report(snapshot_dir: Path) -> list[str]:
    """Validate ``rejected_superseded_report.json`` against its schema.

    Falls back to a structural check when ``jsonschema`` is unavailable or
    the schema file is missing.
    """
    errors: list[str] = []
    report_path = snapshot_dir / REJECTED_SUPERSEDED_REPORT_FILE
    if not report_path.is_file():
        errors.append(f"missing rejected/superseded report: {report_path}")
        return errors

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {report_path}: {exc}")
        return errors

    if not isinstance(payload, dict):
        errors.append(f"{report_path}: report payload must be a JSON object")
        return errors

    schema_payload: dict[str, Any] | None = None
    if REJECTED_SUPERSEDED_REPORT_SCHEMA.is_file():
        try:
            schema_payload = json.loads(
                REJECTED_SUPERSEDED_REPORT_SCHEMA.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            schema_payload = None

    if schema_payload is not None:
        try:
            import jsonschema

            jsonschema.validate(instance=payload, schema=schema_payload)
        except ImportError:
            for field in REJECTED_SUPERSEDED_REPORT_REQUIRED_FIELDS:
                if field not in payload:
                    errors.append(
                        f"{report_path.name}: missing required field {field!r}"
                    )
            if (
                isinstance(payload.get("schema_version"), str)
                and payload["schema_version"]
                != REJECTED_SUPERSEDED_REPORT_SCHEMA_VERSION
            ):
                errors.append(
                    f"{report_path.name}: schema_version must be "
                    f"{REJECTED_SUPERSEDED_REPORT_SCHEMA_VERSION!r}, got "
                    f"{payload['schema_version']!r}"
                )
        except jsonschema.ValidationError as exc:
            errors.append(f"{report_path.name}: schema validation error: {exc.message}")
    else:
        for field in REJECTED_SUPERSEDED_REPORT_REQUIRED_FIELDS:
            if field not in payload:
                errors.append(f"{report_path.name}: missing required field {field!r}")
        if (
            isinstance(payload.get("schema_version"), str)
            and payload["schema_version"] != REJECTED_SUPERSEDED_REPORT_SCHEMA_VERSION
        ):
            errors.append(
                f"{report_path.name}: schema_version must be "
                f"{REJECTED_SUPERSEDED_REPORT_SCHEMA_VERSION!r}, got "
                f"{payload['schema_version']!r}"
            )

    return errors


def _full_hex_git_commit(value: Any) -> str | None:
    """Return ``value`` if it is a full 40-hex git commit string, else ``None``.

    Only full 40-hex values are accepted — no 7-char prefixes and no incidental
    hex strings (engine_version, URLs, IDs), which avoids spurious failures.
    """
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    return None


def _workload_name(value: Any) -> str | None:
    """Resolve the workload label from a public snapshot entry/scope.

    ``workload`` may be a bare string (e.g. ``leaderboard_compare.json`` scopes)
    or an object with a ``name`` field (e.g. leaderboard entry records).
    """
    if isinstance(value, dict):
        name = value.get("name")
        return name if isinstance(name, str) else None
    if isinstance(value, str):
        return value
    return None


def _commit_pairs_in_payload(name: str, payload: Any) -> set[tuple[str | None, str]]:
    """Collect ``(workload, git_commit)`` pairs from one public snapshot payload.

    Handles the two payload shapes synced to the public website:

    - ``leaderboard_compare.json``: a dict with ``scopes[]`` where each scope
      carries ``scope.workload`` and ``latest``/``previous`` ``git_commit``.
    - Entry-list files (leaderboard_single/multi.json): array of entries with
      ``workload.name`` and ``metadata.git_commit``.
    """
    pairs: set[tuple[str | None, str]] = set()
    if name == "leaderboard_compare.json":
        scopes = payload.get("scopes") if isinstance(payload, dict) else None
        for scope_block in scopes or []:
            if not isinstance(scope_block, dict):
                continue
            workload = _workload_name(scope_block.get("scope"))
            for slot in ("latest", "previous"):
                block = scope_block.get(slot)
                if not isinstance(block, dict):
                    continue
                commit = _full_hex_git_commit(block.get("git_commit"))
                if commit is not None:
                    pairs.add((workload, commit))
    else:
        for entry in payload if isinstance(payload, list) else []:
            if not isinstance(entry, dict):
                continue
            workload = _workload_name(entry.get("workload"))
            metadata = (
                entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            )
            commit = _full_hex_git_commit(metadata.get("git_commit"))
            if commit is not None:
                pairs.add((workload, commit))
    return pairs


def _public_snapshot_commit_pairs(snapshot_dir: Path) -> set[tuple[str | None, str]]:
    """Collect ``(workload, git_commit)`` pairs referenced by every public
    leaderboard snapshot file, so the quarantine validator can enforce that
    suspect (commit, workload) pairs stay excluded from public output.
    """
    pairs: set[tuple[str | None, str]] = set()
    for name in PUBLIC_COMMIT_SCAN_FILES:
        path = snapshot_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        pairs |= _commit_pairs_in_payload(name, payload)
    return pairs


def validate_quarantine_suspect_entries(snapshot_dir: Path) -> list[str]:
    """Validate the additive ``issue_146_suspect_entries`` audit section of the
    quarantine file against its JSON schema.

    The section is purely additive (it does not change the issue-105
    ``quarantined_entries`` semantics), so an absent section is acceptable.
    When present, every entry must carry full 40-hex commit provenance and a
    retest delta relative to an explicit base commit.
    """
    errors: list[str] = []
    quarantine_path = snapshot_dir / QUARANTINE_FILE
    if not quarantine_path.is_file():
        return errors

    try:
        payload = json.loads(quarantine_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"invalid JSON in {quarantine_path}: {exc}")
        return errors

    if not isinstance(payload, dict):
        errors.append(f"{quarantine_path.name}: payload must be a JSON object")
        return errors

    schema_payload: dict[str, Any] | None = None
    if QUARANTINE_SUSPECT_SCHEMA.is_file():
        try:
            schema_payload = json.loads(
                QUARANTINE_SUSPECT_SCHEMA.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            schema_payload = None

    for key in QUARANTINE_ADDITIVE_SECTIONS:
        section = payload.get(key)
        if section is None:
            continue
        if not isinstance(section, dict):
            errors.append(f"{quarantine_path.name}:{key} must be a JSON object")
            continue
        section_version = section.get("schema_version")
        if section_version != QUARANTINE_SUSPECT_SCHEMA_VERSION:
            errors.append(
                f"{quarantine_path.name}:{key} schema_version must be "
                f"{QUARANTINE_SUSPECT_SCHEMA_VERSION!r}, got {section_version!r}"
            )
        if schema_payload is not None:
            try:
                import jsonschema

                jsonschema.validate(instance=section, schema=schema_payload)
            except ImportError:
                errors.extend(_structural_suspect_checks(section, quarantine_path, key))
            except jsonschema.ValidationError as exc:
                errors.append(
                    f"{quarantine_path.name}:{key} schema validation error: "
                    f"{exc.message}"
                )
        else:
            errors.extend(_structural_suspect_checks(section, quarantine_path, key))

    # Enforce that suspect entries flagged for exclusion never appear in the
    # public leaderboard snapshots (review: "isolation must be consumed"). The
    # match is workload-aware: only the exact (commit, workload) pair is blocked,
    # so the same commit under a non-suspect workload is not a false positive.
    public_pairs = _public_snapshot_commit_pairs(snapshot_dir)
    for key in QUARANTINE_ADDITIVE_SECTIONS:
        section = payload.get(key)
        if not isinstance(section, dict):
            continue
        section_action = section.get("action")
        for entry in section.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            status = entry.get("status")
            if status != "invalid-suspect-noise" and section_action != "exclude":
                continue
            commit = _full_hex_git_commit(entry.get("git_commit"))
            workload = _workload_name(entry.get("workload"))
            if commit is None:
                continue
            if (workload, commit) in public_pairs:
                errors.append(
                    f"{quarantine_path.name}:{key}: suspect entry "
                    f"{workload!r} @ {commit[:12]} must stay excluded from "
                    f"public snapshots but was found present"
                )
    return errors


def _structural_suspect_checks(
    section: dict[str, Any], quarantine_path: Path, key: str
) -> list[str]:
    """Structural fallback checks used when ``jsonschema`` is unavailable."""
    errors: list[str] = []
    for required in ("conclusion", "action", "note", "entries"):
        if required not in section:
            errors.append(
                f"{quarantine_path.name}:{key} missing required field {required!r}"
            )
    entries = section.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append(f"{quarantine_path.name}:{key} entries must be a non-empty array")
        return errors
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append(f"{quarantine_path.name}:{key} entry must be an object")
            continue
        for field in ("git_commit", "retest_base_commit"):
            value = entry.get(field)
            if not isinstance(value, str) or len(value) != 40:
                errors.append(
                    f"{quarantine_path.name}:{key} entry missing 40-hex {field!r}"
                )
        if "retest_delta_vs_base_commit_pct" not in entry:
            errors.append(
                f"{quarantine_path.name}:{key} entry missing "
                "retest_delta_vs_base_commit_pct"
            )
    return errors


def main() -> int:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    errors: list[str] = []

    hash_fingerprints: dict[str, tuple[str, str]] = {}
    for file_name in SNAPSHOT_FILES:
        path = snapshot_dir / file_name
        if not path.is_file():
            errors.append(f"missing snapshot file: {path}")
            continue
        for entry in load_entries(path):
            errors.extend(validate_entry(entry, source=path))
            same_spec = (
                entry.get("same_spec")
                if isinstance(entry.get("same_spec"), dict)
                else {}
            )
            spec_hash = str(same_spec.get("resolved_spec_hash") or "")
            if not spec_hash:
                continue
            entry_id = str(entry.get("entry_id") or "<missing-entry-id>")
            try:
                fingerprint = compute_resolved_spec_hash(same_spec)
            except (TypeError, ValueError) as error:
                errors.append(
                    f"{path.name}:{entry_id}: invalid same_spec parameters: {error}"
                )
                continue
            source_label = f"{path.name}:{entry_id}"
            prior = hash_fingerprints.get(spec_hash)
            if prior is not None and prior[0] != fingerprint:
                errors.append(
                    f"{source_label}: same_spec hash {spec_hash!r} maps to different "
                    f"effective parameters than {prior[1]}"
                )
            else:
                hash_fingerprints[spec_hash] = (fingerprint, source_label)

    errors.extend(validate_rejected_superseded_report(snapshot_dir))
    errors.extend(validate_quarantine_suspect_entries(snapshot_dir))

    if errors:
        print("public leaderboard snapshot validation failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    print("public leaderboard snapshots passed retired-baseline checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
