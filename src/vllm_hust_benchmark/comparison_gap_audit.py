from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "leaderboard-comparison-gap-audit/v1"
EXCLUDED_WORKLOADS: frozenset[str] = frozenset()
PRODUCTION_TRACE_PROFILE = "production-trace"
PRODUCTION_TRACE_ATTESTATION_SCHEMA = "official-baseline-attestation/v1"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _submitted_at(entry: Mapping[str, Any]) -> str:
    return str(_mapping(entry.get("metadata")).get("submitted_at") or "")


def _resolved_value(
    actual: Mapping[str, Any], key: str, *, workload: str
) -> tuple[bool, Any]:
    aliases: dict[str, tuple[str, ...]] = {
        "input_len": ("random_input_len",),
        "output_len": ("random_output_len", "prefix_repetition_output_len"),
    }
    if key == "input_len" and workload.startswith("prefix-repetition"):
        prefix_len = actual.get("prefix_repetition_prefix_len")
        suffix_len = actual.get("prefix_repetition_suffix_len")
        if isinstance(prefix_len, int) and isinstance(suffix_len, int):
            return True, prefix_len + suffix_len
    if key in actual:
        return True, actual[key]
    for alias in aliases.get(key, ()):
        if alias in actual:
            return True, actual[alias]
    return False, None


def _parameter_reasons(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    prefix: str,
    workload: str,
) -> list[str]:
    reasons: list[str] = []
    for key, expected_value in sorted(expected.items()):
        present, actual_value = _resolved_value(actual, key, workload=workload)
        if not present:
            reasons.append(f"{prefix}.{key}:missing")
        elif actual_value != expected_value:
            reasons.append(
                f"{prefix}.{key}:mismatch:{actual_value!r}!={expected_value!r}"
            )
    return reasons


def _successful_repeats(entry: Mapping[str, Any]) -> int:
    metadata = _mapping(entry.get("metadata"))
    attestation = _mapping(metadata.get("verification_attestation"))
    value = attestation.get("successful_repeats")
    if isinstance(value, int):
        return value
    aggregate = _mapping(entry.get("canonical_aggregate"))
    for key in ("n", "repeat_count", "successful_repeats"):
        value = aggregate.get(key)
        if isinstance(value, int):
            return value
    return 0


def _entry_reasons(
    entry: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    engine: str,
    current_core_head: str | None,
    current_plugin_head: str | None,
) -> list[str]:
    reasons: list[str] = []
    metadata = _mapping(entry.get("metadata"))
    same_spec = _mapping(entry.get("same_spec"))
    workload = _mapping(entry.get("workload"))
    target_workload = _mapping(target.get("workload"))
    target_hardware = _mapping(target.get("hardware"))
    target_model = _mapping(target.get("model"))
    production_trace = target.get("profile") == PRODUCTION_TRACE_PROFILE
    attestation = _mapping(metadata.get("verification_attestation"))

    if metadata.get("verified") is not True:
        reasons.append("verified-attestation-missing")
    if metadata.get("target_id") != target.get("target_id"):
        reasons.append("target-binding-missing-or-mismatched")
    if metadata.get("target_version") != target.get("target_version"):
        reasons.append("target-version-missing-or-mismatched")
    if (
        not production_trace
        and metadata.get("workload_config_contract") != "explicit-effective/v1"
    ):
        reasons.append("explicit-effective-contract-missing")
    if not production_trace and not metadata.get("reproducible_cmd"):
        reasons.append("reproducible-command-missing")
    if production_trace and not (
        attestation.get("schema_version") == PRODUCTION_TRACE_ATTESTATION_SCHEMA
        and attestation.get("evidence") == "repeat_suite.json"
    ):
        reasons.append("production-trace-attestation-missing-or-invalid")

    expected_identity = {
        "same_spec.spec_id": (target.get("target_id"), same_spec.get("spec_id")),
        "workload.name": (target_workload.get("name"), workload.get("name")),
        "hardware.chip_count": (
            target_hardware.get("chip_count"),
            _mapping(entry.get("hardware")).get("chip_count"),
        ),
        "model.id": (
            target_model.get("id"),
            same_spec.get("model")
            or _mapping(entry.get("model")).get("repo_id")
            or _mapping(entry.get("model")).get("name"),
        ),
        "model.precision": (
            target_model.get("precision"),
            same_spec.get("model_precision")
            or _mapping(entry.get("model")).get("precision"),
        ),
    }
    for field, (expected, actual) in expected_identity.items():
        if expected != actual:
            reasons.append(f"{field}:mismatch:{actual!r}!={expected!r}")

    workload_name = str(target_workload.get("name") or "")
    reasons.extend(
        _parameter_reasons(
            _mapping(target.get("server_parameters")),
            _mapping(same_spec.get("resolved_server_parameters")),
            prefix="server_parameters",
            workload=workload_name,
        )
    )
    reasons.extend(
        _parameter_reasons(
            _mapping(target_workload.get("client_parameters")),
            _mapping(same_spec.get("resolved_client_parameters")),
            prefix="client_parameters",
            workload=workload_name,
        )
    )
    if not same_spec.get("resolved_spec_hash"):
        reasons.append("resolved-spec-hash-missing")
    if _successful_repeats(entry) < 3:
        reasons.append("three-successful-repeats-missing")

    metrics = _mapping(entry.get("metrics"))
    if metrics.get("error_rate") not in (0, 0.0):
        reasons.append("nonzero-or-missing-error-rate")
    if not production_trace:
        peak_mem = metrics.get("peak_mem_mb")
        if not isinstance(peak_mem, (int, float)) or peak_mem <= 0:
            reasons.append("peak-memory-unmeasured")

        environment = _mapping(entry.get("environment"))
        for field in ("cann_version", "driver_version", "pytorch_version"):
            if not environment.get(field):
                reasons.append(f"runtime-environment.{field}:missing")

    provenance = _mapping(metadata.get("runtime_provenance"))
    engine_provenance = _mapping(provenance.get("engine"))
    plugin_provenance = _mapping(provenance.get("plugin"))
    if not engine_provenance.get("commit"):
        reasons.append("runtime-provenance.engine-commit:missing")
    if not plugin_provenance.get("commit"):
        reasons.append("runtime-provenance.plugin-commit:missing")

    if engine == "vllm-hust":
        if current_core_head and engine_provenance.get("commit") != current_core_head:
            reasons.append("current-core-head-stale")
        if current_plugin_head and plugin_provenance.get("commit") != current_plugin_head:
            reasons.append("current-plugin-head-stale")
    return sorted(set(reasons))


def _entry_summary(
    entry: Mapping[str, Any] | None,
    target: Mapping[str, Any],
    *,
    engine: str,
    current_core_head: str | None,
    current_plugin_head: str | None,
) -> dict[str, Any] | None:
    if entry is None:
        return None
    reasons = _entry_reasons(
        entry,
        target,
        engine=engine,
        current_core_head=current_core_head,
        current_plugin_head=current_plugin_head,
    )
    same_spec = _mapping(entry.get("same_spec"))
    metadata = _mapping(entry.get("metadata"))
    return {
        "entry_id": entry.get("entry_id"),
        "submitted_at": metadata.get("submitted_at"),
        "git_commit": metadata.get("git_commit"),
        "resolved_spec_hash": same_spec.get("resolved_spec_hash"),
        "successful_repeats": _successful_repeats(entry),
        "eligible": not reasons,
        "reasons": reasons,
    }


def build_comparison_gap_audit(
    repo_root: Path,
    *,
    generated_at: str | None = None,
    current_core_head: str | None = None,
    current_plugin_head: str | None = None,
) -> dict[str, Any]:
    registry_path = repo_root / "leaderboard-data" / "official-targets.json"
    registry = _load_json(registry_path)
    snapshot_dir = repo_root / "leaderboard-data" / "snapshots"
    entries: list[dict[str, Any]] = []
    for name in ("leaderboard_single.json", "leaderboard_multi.json"):
        payload = _load_json(snapshot_dir / name)
        if not isinstance(payload, list):
            raise ValueError(f"snapshot must be an array: {name}")
        entries.extend(item for item in payload if isinstance(item, dict))

    targets = [
        item
        for item in registry.get("targets", [])
        if isinstance(item, dict)
        and item.get("status") == "active"
        and item.get("intended_use") == "public-leaderboard"
        and str(_mapping(item.get("workload")).get("name") or "").lower()
        not in EXCLUDED_WORKLOADS
    ]
    records: list[dict[str, Any]] = []
    rerun_queue: list[dict[str, Any]] = []
    for target in sorted(targets, key=lambda item: str(item.get("target_id") or "")):
        target_id = str(target.get("target_id") or "")
        candidates = [
            entry
            for entry in entries
            if str(_mapping(entry.get("same_spec")).get("spec_id") or "")
            == target_id
        ]
        by_engine = {
            engine: sorted(
                [entry for entry in candidates if entry.get("engine") == engine],
                key=_submitted_at,
                reverse=True,
            )
            for engine in ("vllm", "vllm-hust")
        }
        baseline_entry = by_engine["vllm"][0] if by_engine["vllm"] else None
        current_entry = by_engine["vllm-hust"][0] if by_engine["vllm-hust"] else None
        baseline = _entry_summary(
            baseline_entry,
            target,
            engine="vllm",
            current_core_head=current_core_head,
            current_plugin_head=current_plugin_head,
        )
        current = _entry_summary(
            current_entry,
            target,
            engine="vllm-hust",
            current_core_head=current_core_head,
            current_plugin_head=current_plugin_head,
        )
        hash_match = bool(
            baseline
            and current
            and baseline.get("resolved_spec_hash")
            and baseline.get("resolved_spec_hash") == current.get("resolved_spec_hash")
        )
        ready = bool(
            baseline
            and current
            and baseline.get("eligible")
            and current.get("eligible")
            and hash_match
        )
        required_sides: list[str] = []
        if not baseline or not baseline.get("eligible"):
            required_sides.append("vllm")
        if not current or not current.get("eligible"):
            required_sides.append("vllm-hust")
        if not required_sides and not hash_match:
            required_sides.append("vllm-hust")
        status = "ready" if ready else "rerun-required"
        record = {
            "target_id": target_id,
            "target_version": target.get("target_version"),
            "workload": _mapping(target.get("workload")).get("name"),
            "source_spec": _mapping(target.get("source_spec")).get("path"),
            "status": status,
            "hash_match": hash_match,
            "baseline": baseline,
            "current": current,
            "required_sides": required_sides,
        }
        records.append(record)
        for side in required_sides:
            selected = baseline if side == "vllm" else current
            reasons = ["entry-missing"] if selected is None else list(selected["reasons"])
            if selected and selected.get("eligible") and not hash_match:
                reasons.append("cross-engine-resolved-spec-hash-mismatch")
            rerun_queue.append(
                {
                    "target_id": target_id,
                    "workload": record["workload"],
                    "engine": side,
                    "source_spec": record["source_spec"],
                    "reasons": sorted(set(reasons)),
                    "repeat_count": 3,
                    "min_successful_repeats": 3,
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "registry": {
            "version": registry.get("registry_version"),
            "sha256": _sha256(registry_path),
        },
        "policy": {
            "excluded_workloads": sorted(EXCLUDED_WORKLOADS),
            "minimum_successful_repeats": 3,
            "require_verified": True,
            "require_exact_resolved_spec_hash": True,
            "require_current_heads": bool(current_core_head or current_plugin_head),
            "require_runtime_environment": [
                "cann_version",
                "driver_version",
                "pytorch_version",
            ],
            "require_measured_peak_memory": True,
            "production_trace_attestation_substitutes_generic_runtime_fields": True,
        },
        "current_heads": {
            "vllm_hust": current_core_head,
            "vllm_ascend_hust": current_plugin_head,
        },
        "summary": {
            "target_count": len(records),
            "ready_pair_count": sum(item["status"] == "ready" for item in records),
            "rerun_target_count": sum(
                item["status"] == "rerun-required" for item in records
            ),
            "rerun_job_count": len(rerun_queue),
        },
        "records": records,
        "rerun_queue": rerun_queue,
    }
