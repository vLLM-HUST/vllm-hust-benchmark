from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "official-baseline-recovery-audit/v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _nested(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return {}
        current = current.get(key)
    return current if isinstance(current, Mapping) else {}


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


def _parameter_mismatches(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    prefix: str,
    workload: str,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for key, expected_value in sorted(expected.items()):
        present, actual_value = _resolved_value(actual, key, workload=workload)
        if not present or actual_value != expected_value:
            mismatches.append(
                {
                    "field": f"{prefix}.{key}",
                    "expected": expected_value,
                    "actual": actual_value if present else None,
                    "kind": "value-mismatch" if present else "missing",
                }
            )
    return mismatches


def _identity_mismatches(
    entry: Mapping[str, Any], target: Mapping[str, Any]
) -> list[dict[str, Any]]:
    same_spec = _nested(entry, "same_spec")
    hardware = _nested(entry, "hardware")
    model = _nested(entry, "model")
    target_hardware = _nested(target, "hardware")
    target_model = _nested(target, "model")
    expected_actual = (
        ("engine", _nested(target, "baseline_runtime").get("engine"), entry.get("engine")),
        (
            "engine_version",
            _nested(target, "baseline_runtime").get("engine_version"),
            entry.get("engine_version"),
        ),
        ("same_spec.spec_id", target.get("target_id"), same_spec.get("spec_id")),
        ("hardware.vendor", target_hardware.get("vendor"), hardware.get("vendor")),
        (
            "hardware.chip_model",
            target_hardware.get("chip_model"),
            hardware.get("chip_model"),
        ),
        (
            "hardware.chip_count",
            target_hardware.get("chip_count"),
            hardware.get("chip_count"),
        ),
        (
            "same_spec.node_count",
            target_hardware.get("node_count"),
            same_spec.get("node_count"),
        ),
        (
            "model.id",
            target_model.get("id"),
            same_spec.get("model") or model.get("repo_id") or model.get("name"),
        ),
        (
            "model.parameters",
            target_model.get("parameters"),
            same_spec.get("model_parameters") or model.get("parameters"),
        ),
        (
            "model.precision",
            target_model.get("precision"),
            same_spec.get("model_precision") or model.get("precision"),
        ),
    )
    return [
        {
            "field": field,
            "expected": expected,
            "actual": actual,
            "kind": "missing" if actual is None else "value-mismatch",
        }
        for field, expected, actual in expected_actual
        if expected != actual
    ]


def _manifest_evidence(
    submission_dir: Path, entry: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    artifact_path = submission_dir / "run_leaderboard.json"
    manifest_path = submission_dir / "leaderboard_manifest.json"
    evidence = {
        "artifact": artifact_path.name,
        "artifact_sha256": _sha256(artifact_path),
        "manifest": manifest_path.name if manifest_path.is_file() else None,
        "manifest_sha256": _sha256(manifest_path) if manifest_path.is_file() else None,
        "referenced_by_manifest": False,
        "independent_files": [],
    }
    failures: list[str] = []
    if not manifest_path.is_file():
        failures.append("manifest-missing")
        return evidence, failures
    try:
        manifest = _load_object(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        failures.append("manifest-invalid")
        return evidence, failures
    idempotency_key = _nested(entry, "metadata").get("idempotency_key")
    for item in manifest.get("entries", []):
        if not isinstance(item, Mapping):
            continue
        if item.get("leaderboard_artifact") == artifact_path.name and item.get(
            "idempotency_key"
        ) == idempotency_key:
            evidence["referenced_by_manifest"] = True
            break
    if not evidence["referenced_by_manifest"]:
        failures.append("manifest-does-not-bind-artifact")

    independent = sorted(
        path.name
        for path in submission_dir.iterdir()
        if path.is_file()
        and path.name not in {artifact_path.name, manifest_path.name, "STATUS"}
    )
    evidence["independent_files"] = independent
    return evidence, failures


def audit_submission(
    submission_dir: Path,
    entry: Mapping[str, Any],
    target: Mapping[str, Any] | None,
    *,
    registry_sha256: str,
) -> dict[str, Any]:
    metadata = _nested(entry, "metadata")
    target_id = str(_nested(entry, "same_spec").get("spec_id") or "")
    evidence, manifest_failures = _manifest_evidence(submission_dir, entry)
    reasons = list(manifest_failures)
    mismatches: list[dict[str, Any]] = []
    active_public = bool(
        target
        and target.get("status") == "active"
        and target.get("intended_use") == "public-leaderboard"
    )
    if target is None:
        reasons.append("target-not-in-registry")
    elif not active_public:
        reasons.append("target-not-active-public")
    else:
        workload = str(_nested(target, "workload").get("name") or "")
        mismatches.extend(_identity_mismatches(entry, target))
        mismatches.extend(
            _parameter_mismatches(
                _nested(target, "server_parameters"),
                _nested(entry, "same_spec", "resolved_server_parameters"),
                prefix="server_parameters",
                workload=workload,
            )
        )
        mismatches.extend(
            _parameter_mismatches(
                _nested(target, "workload", "client_parameters"),
                _nested(entry, "same_spec", "resolved_client_parameters"),
                prefix="client_parameters",
                workload=workload,
            )
        )
        source = _nested(target, "source_spec")
        source_path = submission_dir.parents[1] / str(source.get("path") or "")
        if not source_path.is_file() or _sha256(source_path) != source.get("sha256"):
            mismatches.append(
                {
                    "field": "source_spec.sha256",
                    "expected": source.get("sha256"),
                    "actual": _sha256(source_path) if source_path.is_file() else None,
                    "kind": "value-mismatch" if source_path.is_file() else "missing",
                }
            )
        if mismatches:
            reasons.append("exact-target-mismatch")

    if metadata.get("verified") is not True:
        reasons.append("verified-attestation-missing")
    if metadata.get("target_id") != target_id:
        reasons.append("target-binding-missing-or-mismatched")
    target_version_matches = bool(
        target and metadata.get("target_version") == target.get("target_version")
    )
    if target and not target_version_matches:
        reasons.append("target-version-missing-or-mismatched")
    registry_hash_matches = metadata.get("target_registry_sha256") == registry_sha256
    # A registry release can add an unrelated target while leaving this target's
    # contract unchanged. Preserve a historical attestation when its per-target
    # version still matches and the current exact-target comparison succeeds.
    historical_exact_binding = bool(
        target_version_matches
        and metadata.get("target_registry_sha256")
        and not mismatches
        and metadata.get("verified") is True
    )
    if not registry_hash_matches and not historical_exact_binding:
        reasons.append("target-registry-hash-missing-or-mismatched")
    evidence["registry_binding"] = (
        "current-registry"
        if registry_hash_matches
        else "historical-registry-exact-target"
        if historical_exact_binding
        else "unverified"
    )

    recoverable = active_public and not reasons and not mismatches
    disposition = (
        "recoverable"
        if recoverable
        else "rerun-required"
        if active_public
        else "not-public-candidate"
    )
    return {
        "submission": submission_dir.name,
        "entry_id": entry.get("entry_id"),
        "target_id": target_id,
        "target_status": target.get("status") if target else None,
        "target_intended_use": target.get("intended_use") if target else None,
        "disposition": disposition,
        "reasons": sorted(set(reasons)),
        "exact_mismatches": mismatches,
        "evidence": evidence,
        "rerun_spec": target.get("source_spec", {}).get("path")
        if active_public and target
        else None,
    }


def build_recovery_audit(
    repo_root: Path, *, generated_at: str | None = None
) -> dict[str, Any]:
    registry_path = repo_root / "leaderboard-data" / "official-targets.json"
    checksum_path = repo_root / "leaderboard-data" / "official-targets.sha256"
    registry_sha256 = _sha256(registry_path)
    declared = checksum_path.read_text(encoding="utf-8").split()[0]
    if declared != registry_sha256:
        raise ValueError(
            f"official target registry checksum mismatch: {declared} != {registry_sha256}"
        )
    registry = _load_object(registry_path)
    targets = {
        str(target["target_id"]): target
        for target in registry.get("targets", [])
        if isinstance(target, Mapping) and target.get("target_id")
    }
    records: list[dict[str, Any]] = []
    for artifact in sorted(
        (repo_root / "submissions").glob(
            "official-ascend-jan-2026-v0.18.0-*/run_leaderboard.json"
        )
    ):
        entry = _load_object(artifact)
        target_id = str(_nested(entry, "same_spec").get("spec_id") or "")
        records.append(
            audit_submission(
                artifact.parent,
                entry,
                targets.get(target_id),
                registry_sha256=registry_sha256,
            )
        )
    active_records = [
        record
        for record in records
        if record["target_status"] == "active"
        and record["target_intended_use"] == "public-leaderboard"
    ]
    rerun_specs = sorted(
        {
            str(record["rerun_spec"])
            for record in active_records
            if record["disposition"] == "rerun-required" and record["rerun_spec"]
        }
    )
    rerun_env = [
        "REPEAT_COUNT=3",
        "MIN_SUCCESSFUL_REPEATS=3",
        "MATRIX_RESULT_ROOT=.benchmarks/official-baseline-recovery-runs",
        "EXISTING_CANONICAL_SUBMISSIONS_ROOT=.benchmarks/official-baseline-recovery-empty",
        "CANONICAL_SUBMISSIONS_ROOT=.benchmarks/official-baseline-recovery-staged",
    ]
    rerun_args = [
        "env",
        *rerun_env,
        "bash",
        "scripts/run-official-ascend-goal-baseline-matrix.sh",
        *rerun_specs,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "registry": {
            "version": registry.get("registry_version"),
            "sha256": registry_sha256,
        },
        "summary": {
            "scanned": len(records),
            "active_public_candidates": len(active_records),
            "recoverable": sum(r["disposition"] == "recoverable" for r in records),
            "rerun_required": sum(
                r["disposition"] == "rerun-required" for r in active_records
            ),
            "provisional_or_specialty": len(records) - len(active_records),
        },
        "records": records,
        "rerun_specs": rerun_specs,
        "rerun_args": rerun_args,
        "rerun_command": " ".join(rerun_args),
    }
