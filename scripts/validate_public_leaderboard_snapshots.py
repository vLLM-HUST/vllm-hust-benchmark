#!/usr/bin/env python3
"""Validate that public leaderboard snapshots do not include retired baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_hust_benchmark.same_spec import compute_resolved_spec_hash
from vllm_hust_benchmark.workload_config_contract import (
    WORKLOAD_CONFIG_CONTRACT_VERSION,
    requires_workload_config_contract,
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
COMPARE_SNAPSHOT_FILE = "leaderboard_compare.json"
OFFICIAL_TARGET_REGISTRY = REPO_ROOT / "leaderboard-data" / "official-targets.json"
OFFICIAL_TARGET_REGISTRY_CHECKSUM = (
    REPO_ROOT / "leaderboard-data" / "official-targets.sha256"
)
PRODUCTION_TRACE_PROFILE = "production-trace"
PRODUCTION_TRACE_ATTESTATION_SCHEMA = "official-baseline-attestation/v1"
PRODUCTION_TRACE_MINIMUM_REPEATS = 3


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


def is_attested_production_trace_baseline(entry: dict[str, Any]) -> bool:
    """Allow the separately pinned production-trace baseline family fail closed."""
    if not OFFICIAL_TARGET_REGISTRY.is_file() or not OFFICIAL_TARGET_REGISTRY_CHECKSUM.is_file():
        return False
    registry_bytes = OFFICIAL_TARGET_REGISTRY.read_bytes()
    registry_sha256 = hashlib.sha256(registry_bytes).hexdigest()
    declared_sha256 = OFFICIAL_TARGET_REGISTRY_CHECKSUM.read_text(
        encoding="utf-8"
    ).split()[0]
    if registry_sha256 != declared_sha256:
        return False

    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    same_spec = entry.get("same_spec") if isinstance(entry.get("same_spec"), dict) else {}
    target_id = str(same_spec.get("spec_id") or "")
    attestation = (
        metadata.get("verification_attestation")
        if isinstance(metadata.get("verification_attestation"), dict)
        else {}
    )
    if not (
        entry.get("engine") == PUBLIC_BASELINE_ENGINE
        and metadata.get("verified") is True
        and metadata.get("target_id") == target_id
        and metadata.get("profile_id") == PRODUCTION_TRACE_PROFILE
        and metadata.get("target_registry_sha256") == registry_sha256
        and attestation.get("schema_version") == PRODUCTION_TRACE_ATTESTATION_SCHEMA
        and int(attestation.get("successful_repeats") or 0)
        >= PRODUCTION_TRACE_MINIMUM_REPEATS
    ):
        return False

    registry = json.loads(registry_bytes)
    target = next(
        (
            item
            for item in registry.get("targets", [])
            if isinstance(item, dict) and item.get("target_id") == target_id
        ),
        None,
    )
    if not target or not (
        target.get("profile") == PRODUCTION_TRACE_PROFILE
        and target.get("status") == "active"
        and target.get("intended_use") == "public-leaderboard"
        and metadata.get("target_version") == target.get("target_version")
    ):
        return False

    model = entry.get("model") if isinstance(entry.get("model"), dict) else {}
    hardware = entry.get("hardware") if isinstance(entry.get("hardware"), dict) else {}
    workload = entry.get("workload") if isinstance(entry.get("workload"), dict) else {}
    runtime = target.get("baseline_runtime") or {}
    provenance = metadata.get("runtime_provenance") or {}
    return bool(
        entry.get("engine_version") == runtime.get("engine_version")
        and workload.get("name") == (target.get("workload") or {}).get("name")
        and (model.get("name") or model.get("repo_id"))
        == (target.get("model") or {}).get("id")
        and model.get("precision") == (target.get("model") or {}).get("precision")
        and hardware.get("chip_model") == (target.get("hardware") or {}).get("chip_model")
        and hardware.get("chip_count") == (target.get("hardware") or {}).get("chip_count")
        and (provenance.get("engine") or {}).get("commit") == runtime.get("core_commit")
        and (provenance.get("plugin") or {}).get("commit") == runtime.get("backend_commit")
    )


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
    production_trace_baseline = is_attested_production_trace_baseline(entry)

    if (
        engine == PUBLIC_BASELINE_ENGINE
        and engine_version != PUBLIC_BASELINE_VERSION
        and not production_trace_baseline
    ):
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
    if (
        entry_precision_value in RETIRED_PUBLIC_PRECISIONS
        and not production_trace_baseline
    ):
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
    ):
        errors.append(
            f"{source.name}:{entry_id}: public vllm-hust official workload "
            f"{workload!r} must use official v0.18.0 same_spec, got {spec_id!r}"
        )

    if spec_id.startswith(OFFICIAL_V0180_SPEC_PREFIX):
        gpu_memory_utilization = parse_optional_float(
            resolved_server_parameters.get("gpu_memory_utilization")
        )
        if (
            gpu_memory_utilization is not None
            and gpu_memory_utilization != DEFAULT_PUBLIC_GPU_MEMORY_UTILIZATION
        ):
            errors.append(
                f"{source.name}:{entry_id}: official public snapshot must not "
                f"publish non-default gpu_memory_utilization="
                f"{resolved_server_parameters.get('gpu_memory_utilization')!r}; "
                f"rerun with {DEFAULT_PUBLIC_GPU_MEMORY_UTILIZATION} or mark the "
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
    if requires_workload_config_contract(entry) or contract_version:
        for message in validate_explicit_workload_config(entry):
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


def validate_compare_snapshot(
    snapshot_dir: Path,
    entries_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    path = snapshot_dir / COMPARE_SNAPSHOT_FILE
    if not path.is_file():
        return [f"missing compare snapshot: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"invalid JSON in {path}: {exc}"]
    if not isinstance(payload, dict):
        return [f"{path}: compare snapshot payload must be a JSON object"]

    pairs = payload.get("preferred_pairs")
    if not isinstance(pairs, list):
        return [f"{path.name}: preferred_pairs must be an array"]
    declared_count = payload.get("preferred_pair_count")
    if declared_count != len(pairs):
        errors.append(
            f"{path.name}: preferred_pair_count={declared_count!r} does not match "
            f"preferred_pairs length={len(pairs)}"
        )

    for index, item in enumerate(pairs):
        pair = item.get("preferred_pair") if isinstance(item, dict) else None
        if not isinstance(pair, dict):
            errors.append(f"{path.name}: preferred_pairs[{index}] missing preferred_pair")
            continue
        summaries: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            summary = pair.get(side)
            if not isinstance(summary, dict):
                errors.append(
                    f"{path.name}: preferred_pairs[{index}].preferred_pair.{side} "
                    "must be an object"
                )
                continue
            summaries[side] = summary
            entry_id = str(summary.get("entry_id") or "")
            source_entry = entries_by_id.get(entry_id)
            if source_entry is None:
                errors.append(
                    f"{path.name}: preferred_pairs[{index}] {side} references "
                    f"unknown entry_id {entry_id!r}"
                )
                continue
            summary_hash = str(
                ((summary.get("same_spec") or {}).get("resolved_spec_hash") or "")
            )
            source_hash = str(
                ((source_entry.get("same_spec") or {}).get("resolved_spec_hash") or "")
            )
            if not summary_hash or summary_hash != source_hash:
                errors.append(
                    f"{path.name}: preferred_pairs[{index}] {side} hash "
                    f"{summary_hash!r} does not match source entry {entry_id!r} "
                    f"hash {source_hash!r}"
                )
        if set(summaries) != {"left", "right"}:
            continue
        left_hash = str(
            ((summaries["left"].get("same_spec") or {}).get("resolved_spec_hash") or "")
        )
        right_hash = str(
            ((summaries["right"].get("same_spec") or {}).get("resolved_spec_hash") or "")
        )
        if not left_hash or left_hash != right_hash:
            errors.append(
                f"{path.name}: preferred_pairs[{index}] resolved_spec_hash mismatch: "
                f"left={left_hash!r} right={right_hash!r}"
            )
    return errors


def main() -> int:
    args = parse_args()
    snapshot_dir = Path(args.snapshot_dir)
    errors: list[str] = []
    entries_by_id: dict[str, dict[str, Any]] = {}

    hash_fingerprints: dict[str, tuple[str, str]] = {}
    for file_name in SNAPSHOT_FILES:
        path = snapshot_dir / file_name
        if not path.is_file():
            errors.append(f"missing snapshot file: {path}")
            continue
        for entry in load_entries(path):
            entry_id = str(entry.get("entry_id") or "")
            if entry_id:
                entries_by_id[entry_id] = entry
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

    errors.extend(validate_compare_snapshot(snapshot_dir, entries_by_id))
    errors.extend(validate_rejected_superseded_report(snapshot_dir))

    if errors:
        print("public leaderboard snapshot validation failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    print("public leaderboard snapshots passed retired-baseline checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
