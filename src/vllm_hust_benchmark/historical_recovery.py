"""Recover auditable historical leaderboard entries without weakening admission.

The public leaderboard admission gate is intentionally fail-closed.  This module
builds a separate historical projection from archived/raw submissions so useful
trend data is not lost merely because newer evidence fields did not exist when a
run was recorded.  Raw artifacts are never modified and missing measurements are
never invented.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.same_spec import compute_resolved_spec_hash

SCHEMA_VERSION = "historical-leaderboard-recovery/v1"
SHA40_LENGTH = 40
RETIRED_PATH_PREFIX = "archive/pre-v0.18.0/"
PRIMARY_METRICS = ("throughput_tps", "ttft_ms", "tbt_ms")


@dataclass(frozen=True)
class RecoveryDecision:
    source_path: str
    disposition: str
    reasons: tuple[str, ...]
    entry: dict[str, Any] | None = None
    selection_key: str | None = None
    quality_score: tuple[int, ...] = ()


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _sha40(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if len(normalized) != SHA40_LENGTH:
        return None
    if any(character not in "0123456789abcdef" for character in normalized):
        return None
    return normalized


def _resolve_sha(
    value: Any,
    revision_aliases: dict[str, str],
    *,
    source_hint: str = "",
) -> tuple[str | None, bool]:
    normalized = str(value or "").strip().lower()
    exact = _sha40(normalized)
    if exact is not None:
        return exact, False
    matches = {
        full
        for prefix, full in revision_aliases.items()
        if (normalized and (prefix == normalized or full.startswith(normalized)))
        or (not normalized and prefix in source_hint.lower())
    }
    if len(matches) == 1:
        return matches.pop(), True
    return None, False


def _runtime_commits(
    entry: dict[str, Any], revision_aliases: dict[str, str], *, source_hint: str
) -> tuple[str | None, str | None, list[str]]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    provenance = (
        metadata.get("runtime_provenance")
        if isinstance(metadata.get("runtime_provenance"), dict)
        else {}
    )
    engine = (
        provenance.get("engine") if isinstance(provenance.get("engine"), dict) else {}
    )
    plugin = (
        provenance.get("plugin") if isinstance(provenance.get("plugin"), dict) else {}
    )
    engine_commit, engine_inferred = _resolve_sha(
        engine.get("commit") or metadata.get("git_commit"), revision_aliases
    )
    plugin_commit, plugin_inferred = _resolve_sha(
        plugin.get("commit"), revision_aliases, source_hint=source_hint
    )
    inferred: list[str] = []
    if engine_inferred:
        inferred.append("metadata.runtime_provenance.engine.commit")
    if plugin_inferred:
        inferred.append("metadata.runtime_provenance.plugin.commit")
    return engine_commit, plugin_commit, inferred


def _load_registry(registry_path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    targets = payload.get("targets")
    if not isinstance(targets, list):
        raise TypeError(f"{registry_path}: targets must be an array")
    by_id = {
        str(target["target_id"]): target
        for target in targets
        if isinstance(target, dict) and target.get("target_id")
    }
    return by_id, str(payload.get("registry_version") or "")


def _load_revision_aliases(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    aliases = payload.get("aliases")
    if not isinstance(aliases, dict):
        raise TypeError(f"{path}: aliases must be an object")
    resolved: dict[str, str] = {}
    for prefix, full in aliases.items():
        exact = _sha40(full)
        if exact is None or not exact.startswith(str(prefix).lower()):
            raise ValueError(f"{path}: invalid revision alias {prefix!r} -> {full!r}")
        resolved[str(prefix).lower()] = exact
    return resolved


def discover_artifacts(repo_root: Path) -> list[Path]:
    """Return stable, unique raw artifact paths from submissions and archives."""

    paths: set[Path] = set()
    for directory in (repo_root / "submissions", repo_root / "archive"):
        if directory.is_dir():
            paths.update(directory.rglob("run_leaderboard.json"))
    return sorted(paths, key=lambda path: path.relative_to(repo_root).as_posix())


def _load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload if isinstance(payload, list) else [payload]
    return [value for value in values if isinstance(value, dict)]


def _quality_score(
    *,
    method: str,
    entry: dict[str, Any],
    artifact_path: Path,
) -> tuple[int, ...]:
    method_score = {
        "historical-exact-spec": 2,
        "historical-reconstructed-hash": 1,
    }[method]
    return (
        method_score,
        int((artifact_path.parent / "env-manifest.json").is_file()),
        int((artifact_path.parent / "checksums.sha256").is_file()),
    )


def _selection_key(
    entry: dict[str, Any], *, engine_commit: str, plugin_commit: str, spec_hash: str
) -> str:
    identity = {
        "engine_commit": engine_commit,
        "plugin_commit": plugin_commit,
        "resolved_spec_hash": spec_hash,
        "config_type": str(entry.get("config_type") or ""),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def recover_entry(
    entry: dict[str, Any],
    *,
    artifact_path: Path,
    repo_root: Path,
    registry_by_id: dict[str, dict[str, Any]],
    registry_version: str,
    revision_aliases: dict[str, str],
) -> RecoveryDecision:
    source_path = artifact_path.relative_to(repo_root).as_posix()
    if source_path.startswith(RETIRED_PATH_PREFIX):
        return RecoveryDecision(source_path, "rejected", ("retired-pre-v0.18.0",))

    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    if not any(
        _is_finite_number(metrics.get(name)) and float(metrics[name]) > 0
        for name in PRIMARY_METRICS
    ):
        return RecoveryDecision(source_path, "rejected", ("invalid-primary-metrics",))
    error_rate = metrics.get("error_rate")
    if not _is_finite_number(error_rate) or not 0 <= float(error_rate) <= 1:
        return RecoveryDecision(source_path, "rejected", ("invalid-error-rate",))

    engine_commit, plugin_commit, revision_inferences = _runtime_commits(
        entry, revision_aliases, source_hint=source_path
    )
    missing_runtime = []
    if engine_commit is None:
        missing_runtime.append("engine-commit")
    if plugin_commit is None:
        missing_runtime.append("plugin-commit")
    if missing_runtime:
        return RecoveryDecision(source_path, "rejected", tuple(missing_runtime))

    same_spec = entry.get("same_spec")
    if not isinstance(same_spec, dict) or not same_spec:
        return RecoveryDecision(source_path, "rejected", ("same-spec-missing",))
    try:
        recomputed_hash = compute_resolved_spec_hash(same_spec)
    except (TypeError, ValueError) as error:
        return RecoveryDecision(
            source_path,
            "rejected",
            (f"same-spec-incomplete:{error}",),
        )

    recovered = copy.deepcopy(entry)
    recovered_spec = recovered["same_spec"]
    original_hash = str(recovered_spec.get("resolved_spec_hash") or "")
    inferred_fields: list[str] = list(revision_inferences)
    if original_hash == recomputed_hash:
        method = "historical-exact-spec"
    else:
        method = "historical-reconstructed-hash"
        recovered_spec["resolved_spec_hash"] = recomputed_hash
        inferred_fields.append("same_spec.resolved_spec_hash")

    metadata = recovered.setdefault("metadata", {})
    provenance = metadata.setdefault("runtime_provenance", {})
    engine_provenance = provenance.setdefault("engine", {})
    plugin_provenance = provenance.setdefault("plugin", {})
    engine_provenance["commit"] = engine_commit
    plugin_provenance["commit"] = plugin_commit
    spec_id = str(recovered_spec.get("spec_id") or "")
    target = registry_by_id.get(spec_id)
    if target is not None:
        if not metadata.get("target_id"):
            metadata["target_id"] = spec_id
            inferred_fields.append("metadata.target_id")
        if not metadata.get("target_version"):
            metadata["target_version"] = str(
                target.get("target_version") or registry_version
            )
            inferred_fields.append("metadata.target_version")

    key = _selection_key(
        recovered,
        engine_commit=engine_commit,
        plugin_commit=plugin_commit,
        spec_hash=recomputed_hash,
    )
    recovered["historical_recovery"] = {
        "schema_version": SCHEMA_VERSION,
        "recovery_method": method,
        "source_path": source_path,
        "selection_key": key,
        "inferred_fields": sorted(inferred_fields),
        "original_resolved_spec_hash": original_hash
        if original_hash != recomputed_hash
        else None,
        "admitted_for_historical_trend": True,
    }
    score = _quality_score(method=method, entry=recovered, artifact_path=artifact_path)
    return RecoveryDecision(
        source_path,
        "candidate",
        (),
        entry=recovered,
        selection_key=key,
        quality_score=score,
    )


def _candidate_rank(decision: RecoveryDecision) -> tuple[Any, ...]:
    assert decision.entry is not None
    metadata = (
        decision.entry.get("metadata")
        if isinstance(decision.entry.get("metadata"), dict)
        else {}
    )
    # Date/path are deterministic tie-breakers only. Metrics deliberately do
    # not participate, preventing best-of-run selection bias.
    return decision.quality_score + (
        str(metadata.get("submitted_at") or ""),
        decision.source_path,
    )


def build_recovery(
    *,
    repo_root: Path,
    registry_path: Path,
    revision_aliases_path: Path | None = None,
    artifact_paths: Iterable[Path] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    registry_by_id, registry_version = _load_registry(registry_path)
    revision_aliases = _load_revision_aliases(revision_aliases_path)
    paths = (
        list(artifact_paths)
        if artifact_paths is not None
        else discover_artifacts(repo_root)
    )
    decisions: list[RecoveryDecision] = []
    for path in sorted(
        paths, key=lambda value: value.relative_to(repo_root).as_posix()
    ):
        try:
            entries = _load_entries(path)
        except (OSError, json.JSONDecodeError) as error:
            decisions.append(
                RecoveryDecision(
                    path.relative_to(repo_root).as_posix(),
                    "rejected",
                    (f"unreadable-artifact:{error}",),
                )
            )
            continue
        for entry in entries:
            decisions.append(
                recover_entry(
                    entry,
                    artifact_path=path,
                    repo_root=repo_root,
                    registry_by_id=registry_by_id,
                    registry_version=registry_version,
                    revision_aliases=revision_aliases,
                )
            )

    selected_by_key: dict[str, RecoveryDecision] = {}
    for decision in decisions:
        if decision.disposition != "candidate" or decision.selection_key is None:
            continue
        previous = selected_by_key.get(decision.selection_key)
        if previous is None or _candidate_rank(decision) > _candidate_rank(previous):
            selected_by_key[decision.selection_key] = decision

    selected_paths = {decision.source_path for decision in selected_by_key.values()}
    entries = [
        decision.entry
        for decision in sorted(
            selected_by_key.values(), key=lambda item: item.source_path
        )
        if decision.entry is not None
    ]
    rejected = [
        {
            "source_path": decision.source_path,
            "reasons": list(decision.reasons),
        }
        for decision in decisions
        if decision.disposition == "rejected"
    ]
    superseded = [
        {
            "source_path": decision.source_path,
            "selected_source_path": selected_by_key[decision.selection_key].source_path,
            "selection_key": decision.selection_key,
        }
        for decision in decisions
        if decision.disposition == "candidate"
        and decision.selection_key is not None
        and decision.source_path not in selected_paths
    ]
    method_counts: dict[str, int] = {}
    inferred_field_counts: dict[str, int] = {}
    for entry in entries:
        recovery = entry["historical_recovery"]
        method = recovery["recovery_method"]
        method_counts[method] = method_counts.get(method, 0) + 1
        for field in recovery["inferred_fields"]:
            inferred_field_counts[field] = inferred_field_counts.get(field, 0) + 1
    required_experiments = []
    for item in rejected:
        if item["reasons"] == ["retired-pre-v0.18.0"]:
            continue
        decision = next(
            decision
            for decision in decisions
            if decision.source_path == item["source_path"]
            and decision.disposition == "rejected"
        )
        source_entry = next(iter(_load_entries(repo_root / decision.source_path)), {})
        metadata = source_entry.get("metadata") or {}
        provenance = metadata.get("runtime_provenance") or {}
        required_experiments.append(
            {
                "source_path": decision.source_path,
                "workload": str((source_entry.get("workload") or {}).get("name") or ""),
                "spec_id": str(
                    (source_entry.get("same_spec") or {}).get("spec_id") or ""
                ),
                "engine_commit": str(
                    (provenance.get("engine") or {}).get("commit")
                    or metadata.get("git_commit")
                    or ""
                ),
                "plugin_commit": str(
                    (provenance.get("plugin") or {}).get("commit") or ""
                ),
                "missing_or_invalid": item["reasons"],
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "registry_version": registry_version,
        "summary": {
            "source_artifacts": len(paths),
            "evaluated_entries": len(decisions),
            "selected_entries": len(entries),
            "rejected_entries": len(rejected),
            "superseded_entries": len(superseded),
            "spec_recovery_methods": dict(sorted(method_counts.items())),
            "inferred_fields": dict(sorted(inferred_field_counts.items())),
            "required_experiments": len(required_experiments),
        },
        "policy": {
            "raw_artifacts_modified": False,
            "missing_measurements_invented": False,
            "deduplication_uses_metrics": False,
            "formal_admission_gate_unchanged": True,
        },
        "rejected": rejected,
        "superseded": superseded,
        "required_experiments": required_experiments,
    }
    return entries, report


def write_recovery(
    *,
    repo_root: Path,
    output_dir: Path,
    registry_path: Path,
    revision_aliases_path: Path | None = None,
) -> tuple[Path, Path]:
    entries, report = build_recovery(
        repo_root=repo_root,
        registry_path=registry_path,
        revision_aliases_path=revision_aliases_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    entries_path = output_dir / "leaderboard_historical.json"
    report_path = output_dir / "historical_recovery_report.json"
    entries_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return entries_path, report_path
