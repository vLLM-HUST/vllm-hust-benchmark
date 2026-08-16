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
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.model_registry import resolve_model_identity
from vllm_hust_benchmark.same_spec import compute_resolved_spec_hash

SCHEMA_VERSION = "historical-leaderboard-recovery/v1"
SHA40_LENGTH = 40
RETIRED_PATH_PREFIX = "archive/pre-v0.18.0/"
PRIMARY_METRICS = ("throughput_tps", "ttft_ms", "tbt_ms")
ATOMIC_OFFLINE_LATENCY_RULE = "atomic-offline-latency-success/v1"
ONLINE_NO_STREAM_DEFAULT_RULE = "legacy-online-no-stream-default/v1"
EXECUTION_CONTRACT_KEYS = (
    "enable_prefix_caching",
    "gpu_memory_utilization",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "tensor_parallel_size",
)


def _infer_same_spec_defaults(same_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Materialize deterministic legacy CLI defaults in recovered contracts.

    Older ``vllm bench serve`` artifacts omitted ``no_stream`` when the flag was
    not supplied.  The effective value is nevertheless unambiguous: the CLI
    default is ``False``.  Leaving it absent fragments identical experiments
    (including runs from different physical machines) into different hashes.
    """

    client = same_spec.get("resolved_client_parameters")
    if not isinstance(client, dict):
        return []
    scenario = str(same_spec.get("scenario") or "")
    if not scenario.endswith("-online") or "no_stream" in client:
        return []
    client["no_stream"] = False
    return [
        {
            "field": "same_spec.resolved_client_parameters.no_stream",
            "value": False,
            "rule_id": ONLINE_NO_STREAM_DEFAULT_RULE,
            "evidence": {
                "benchmark": "vllm bench serve",
                "cli_semantics": "omitted --no-stream flag resolves to false",
            },
        }
    ]


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(entry: dict[str, Any], path: Path) -> dict[str, str]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    return {
        "entry_id": str(entry.get("entry_id") or ""),
        "idempotency_key": str(metadata.get("idempotency_key") or ""),
        "sha256": _file_sha256(path),
    }


def _is_atomic_offline_latency_success(entry: dict[str, Any]) -> bool:
    """Return whether a successful artifact proves a zero request error rate.

    ``vllm bench latency`` is an in-process, atomic offline benchmark: it emits
    the aggregate latency artifact only after every warmup and measured
    ``LLM.generate`` call returns. Any request failure raises and the command
    exits non-zero instead of publishing the artifact. This rule deliberately
    excludes online workloads and non-historical data sources.
    """

    workload = entry.get("workload") if isinstance(entry.get("workload"), dict) else {}
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    same_spec = (
        entry.get("same_spec") if isinstance(entry.get("same_spec"), dict) else {}
    )
    client = (
        same_spec.get("resolved_client_parameters")
        if isinstance(same_spec.get("resolved_client_parameters"), dict)
        else {}
    )
    return (
        workload.get("name") == "random-latency"
        and same_spec.get("scenario") == "random-latency"
        and metadata.get("data_source") == "real-online-historical-pr-backfill"
        and isinstance(client.get("num_iters"), int)
        and not isinstance(client.get("num_iters"), bool)
        and client["num_iters"] > 0
        and isinstance(client.get("num_iters_warmup"), int)
        and not isinstance(client.get("num_iters_warmup"), bool)
        and client["num_iters_warmup"] >= 0
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
    for directory in (
        repo_root / "submissions",
        repo_root / "reports" / "historical-recovery-evidence",
        repo_root / "archive",
    ):
        if directory.is_dir():
            paths.update(directory.rglob("run_leaderboard.json"))
    return sorted(paths, key=lambda path: path.relative_to(repo_root).as_posix())


def _load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload if isinstance(payload, list) else [payload]
    return [value for value in values if isinstance(value, dict)]


def _load_input_contract(artifact_path: Path) -> dict[str, str] | None:
    """Load a valid frozen-input identity without making its host path material."""
    path = artifact_path.parent / "input_provenance.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != "visionarena-frozen-input/v1":
        return None
    dataset = payload.get("dataset")
    selection = payload.get("selection")
    if not isinstance(dataset, dict) or not isinstance(selection, dict):
        return None
    revision = str(dataset.get("revision") or "").strip().lower()
    content_sha256 = str(selection.get("content_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", content_sha256):
        return None
    return {
        "schema_version": str(payload["schema_version"]),
        "dataset_revision": revision,
        "content_sha256": content_sha256,
    }


def _contract_scalar(value: Any) -> tuple[str, Any]:
    """Normalize JSON scalars for execution-vs-target comparisons."""

    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("number", float(value))
    text = str(value).strip()
    lowered = text.lower()
    if lowered == "true":
        return ("bool", True)
    if lowered == "false":
        return ("bool", False)
    try:
        return ("number", float(text))
    except ValueError:
        return ("text", text)


def _execution_contract_conflicts(
    artifact_path: Path,
    target: dict[str, Any] | None,
    same_spec: dict[str, Any],
) -> tuple[str, ...]:
    """Reject explicit repeat execution evidence that contradicts its target.

    Historical artifacts can omit effective server parameters.  An adjacent
    repeat suite is stronger evidence about what actually ran; when it states a
    value that conflicts with the registered official target, the record is a
    diagnostic run rather than an admissible point for that target.
    """

    if target is None:
        return ()
    registered = target.get("server_parameters")
    if not isinstance(registered, dict):
        return ()

    evidence: list[dict[str, Any]] = []
    resolved_server = same_spec.get("resolved_server_parameters")
    if isinstance(resolved_server, dict):
        evidence.append(resolved_server)
    repeat_suite_path = artifact_path.parent / "repeat_suite.json"
    if repeat_suite_path.is_file():
        try:
            repeat_suite = json.loads(repeat_suite_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            repeat_suite = {}
        execution = repeat_suite.get("execution")
        if isinstance(execution, dict):
            evidence.append(execution)

    conflicts = []
    for execution in evidence:
        for field in EXECUTION_CONTRACT_KEYS:
            if field not in execution or field not in registered:
                continue
            actual = execution[field]
            expected = registered[field]
            reason = f"execution-contract-conflict:{field}:{actual}!={expected}"
            if (
                _contract_scalar(actual) != _contract_scalar(expected)
                and reason not in conflicts
            ):
                conflicts.append(reason)
    return tuple(conflicts)


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
        int(
            (artifact_path.parent / "repeat_suite.json").is_file()
            or int(entry.get("metadata", {}).get("repetitions") or 0) >= 3
        ),
        int((artifact_path.parent / "env-manifest.json").is_file()),
        int((artifact_path.parent / "checksums.sha256").is_file()),
    )


def _selection_key(
    entry: dict[str, Any],
    *,
    engine_commit: str,
    plugin_commit: str,
    spec_hash: str,
    spec_id: str,
    registered_target: bool,
    input_contract: dict[str, str] | None,
) -> str:
    identity = {
        "engine_commit": engine_commit,
        "plugin_commit": plugin_commit,
        "config_type": str(entry.get("config_type") or ""),
    }
    if registered_target:
        identity["spec_id"] = spec_id
        scenario = str(entry.get("same_spec", {}).get("scenario") or "")
        if "visionarena" in scenario.lower():
            identity["input_contract"] = (
                input_contract["content_sha256"]
                if input_contract is not None
                else "unrecorded"
            )
    else:
        identity["resolved_spec_hash"] = spec_hash
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _experiment_key(
    entry: dict[str, Any],
    *,
    revision_aliases: dict[str, str],
    source_hint: str,
) -> str | None:
    """Return the rerun identity even when a measurement is invalid.

    The registered spec may have gained explicit effective defaults since the
    historical run. A real rerun therefore matches on the registered target
    and exact runtime revisions, while its own resolved spec hash remains the
    source of truth for the newly measured point.
    """

    engine_commit, plugin_commit, _ = _runtime_commits(
        entry, revision_aliases, source_hint=source_hint
    )
    same_spec = entry.get("same_spec")
    if (
        engine_commit is None
        or plugin_commit is None
        or not isinstance(same_spec, dict)
    ):
        return None
    spec_id = str(same_spec.get("spec_id") or "")
    if not spec_id:
        return None
    identity = {
        "engine_commit": engine_commit,
        "plugin_commit": plugin_commit,
        "spec_id": spec_id,
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
    derived_atomic_success = (
        not _is_finite_number(error_rate) or not 0 <= float(error_rate) <= 1
    ) and _is_atomic_offline_latency_success(entry)
    if not derived_atomic_success and (
        not _is_finite_number(error_rate) or not 0 <= float(error_rate) <= 1
    ):
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
    recovered = copy.deepcopy(entry)
    if derived_atomic_success:
        recovered["metrics"]["error_rate"] = 0.0
    recovered_spec = recovered["same_spec"]
    original_hash = str(recovered_spec.get("resolved_spec_hash") or "")
    inferred_fields: list[str] = list(revision_inferences)
    measurement_derivations = []
    spec_derivations = _infer_same_spec_defaults(recovered_spec)
    inferred_fields.extend(item["field"] for item in spec_derivations)
    try:
        recomputed_hash = compute_resolved_spec_hash(recovered_spec)
    except (TypeError, ValueError) as error:
        return RecoveryDecision(
            source_path,
            "rejected",
            (f"same-spec-incomplete:{error}",),
        )

    # The resolved same-spec contract is stronger evidence than legacy outer
    # display metadata.  Historical exporters occasionally copied the default
    # 14B Instruct identity into Coder/VL entries even though both the server
    # and client were resolved against the correct target model.  Repair that
    # deterministic metadata mismatch instead of scheduling another experiment.
    spec_model = str(recovered_spec.get("model") or "").strip()
    if spec_model:
        identity = resolve_model_identity(spec_model)
        recovered_model = recovered.setdefault("model", {})
        authoritative_model_fields: dict[str, Any] = {
            "canonical_id": identity.canonical_id,
            "repo_id": identity.repo_id,
            "short_name": identity.short_name,
            "display_name": identity.display_name,
            "name": identity.repo_id,
            "parameters": recovered_spec.get("model_parameters"),
            "precision": recovered_spec.get("model_precision"),
            "quantization": recovered_spec.get("model_quantization") or None,
        }
        for field, value in authoritative_model_fields.items():
            if recovered_model.get(field) != value:
                recovered_model[field] = value
                inferred_fields.append(f"model.{field}")
    if derived_atomic_success:
        inferred_fields.append("metrics.error_rate")
        measurement_derivations.append(
            {
                "field": "metrics.error_rate",
                "value": 0.0,
                "rule_id": ATOMIC_OFFLINE_LATENCY_RULE,
                "evidence": {
                    "benchmark": "vllm bench latency",
                    "success_semantics": (
                        "artifact emitted only after all warmup and measured "
                        "LLM.generate calls return; request failures exit non-zero"
                    ),
                    "num_iters_warmup": recovered_spec["resolved_client_parameters"][
                        "num_iters_warmup"
                    ],
                    "num_iters": recovered_spec["resolved_client_parameters"][
                        "num_iters"
                    ],
                    "original_artifact_identity": _artifact_identity(
                        entry, artifact_path
                    ),
                },
            }
        )
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
    execution_conflicts = _execution_contract_conflicts(
        artifact_path, target, recovered_spec
    )
    if execution_conflicts:
        return RecoveryDecision(source_path, "rejected", execution_conflicts)
    if target is not None:
        if not metadata.get("target_id"):
            metadata["target_id"] = spec_id
            inferred_fields.append("metadata.target_id")
        if not metadata.get("target_version"):
            metadata["target_version"] = str(
                target.get("target_version") or registry_version
            )
            inferred_fields.append("metadata.target_version")

    input_contract = _load_input_contract(artifact_path)
    key = _selection_key(
        recovered,
        engine_commit=engine_commit,
        plugin_commit=plugin_commit,
        spec_hash=recomputed_hash,
        spec_id=spec_id,
        registered_target=target is not None,
        input_contract=input_contract,
    )
    recovery_record = {
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
    if measurement_derivations:
        recovery_record["measurement_derivations"] = measurement_derivations
    if spec_derivations:
        recovery_record["spec_derivations"] = spec_derivations
    if input_contract is not None:
        recovery_record["input_contract"] = input_contract
    recovered["historical_recovery"] = recovery_record
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

    # Prefer a genuine valid replacement over a derivation from its invalid
    # predecessor. Derivation is the fallback for an otherwise unsatisfied
    # experiment, not an extra trend point beside a fresh rerun.
    nonderived_experiments = {
        experiment_key
        for decision in decisions
        if decision.disposition == "candidate"
        and decision.entry is not None
        and not decision.entry.get("historical_recovery", {}).get(
            "measurement_derivations"
        )
        and (
            experiment_key := _experiment_key(
                decision.entry,
                revision_aliases=revision_aliases,
                source_hint=decision.source_path,
            )
        )
        is not None
    }
    for index, decision in enumerate(decisions):
        if decision.disposition != "candidate" or decision.entry is None:
            continue
        if not decision.entry.get("historical_recovery", {}).get(
            "measurement_derivations"
        ):
            continue
        experiment_key = _experiment_key(
            decision.entry,
            revision_aliases=revision_aliases,
            source_hint=decision.source_path,
        )
        if experiment_key in nonderived_experiments:
            decisions[index] = RecoveryDecision(
                decision.source_path,
                "rejected",
                ("invalid-error-rate",),
            )

    selected_by_key: dict[str, RecoveryDecision] = {}
    for decision in decisions:
        if decision.disposition != "candidate" or decision.selection_key is None:
            continue
        previous = selected_by_key.get(decision.selection_key)
        if previous is None or _candidate_rank(decision) > _candidate_rank(previous):
            selected_by_key[decision.selection_key] = decision

    selected_paths = {decision.source_path for decision in selected_by_key.values()}
    selected_by_experiment_key = {
        experiment_key: decision
        for decision in selected_by_key.values()
        if decision.entry is not None
        and (
            experiment_key := _experiment_key(
                decision.entry,
                revision_aliases=revision_aliases,
                source_hint=decision.source_path,
            )
        )
        is not None
    }
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
    satisfied_experiments = [
        {
            "source_path": decision.source_path,
            "replacement_source_path": None,
            "selection_key": decision.selection_key,
            "evidence_kind": "derived-success",
            "derivation_rule": derivations[0]["rule_id"],
            "original_artifact_identity": derivations[0]["evidence"][
                "original_artifact_identity"
            ],
            "replacement_artifact_identity": None,
        }
        for decision in selected_by_key.values()
        if decision.entry is not None
        and (
            derivations := decision.entry.get("historical_recovery", {}).get(
                "measurement_derivations", []
            )
        )
    ]
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
        evidence_key = _experiment_key(
            source_entry,
            revision_aliases=revision_aliases,
            source_hint=decision.source_path,
        )
        if evidence_key is not None and evidence_key in selected_by_experiment_key:
            replacement = selected_by_experiment_key[evidence_key]
            assert replacement.entry is not None
            replacement_path = repo_root / replacement.source_path
            repeat_suite_path = replacement_path.parent / "repeat_suite.json"
            satisfied_experiments.append(
                {
                    "source_path": decision.source_path,
                    "replacement_source_path": replacement.source_path,
                    "selection_key": replacement.selection_key,
                    "evidence_kind": (
                        "existing-strict-repeat-suite"
                        if repeat_suite_path.is_file()
                        else "fresh-rerun"
                    ),
                    "original_artifact_identity": _artifact_identity(
                        source_entry, repo_root / decision.source_path
                    ),
                    "replacement_artifact_identity": _artifact_identity(
                        replacement.entry, replacement_path
                    ),
                }
            )
            continue
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
            "satisfied_experiments": len(satisfied_experiments),
        },
        "policy": {
            "raw_artifacts_modified": False,
            "missing_measurements_invented": False,
            "deduplication_uses_metrics": False,
            "physical_machine_partitions_trend_identity": False,
            "same_chip_cross_machine_comparable": True,
            "explicit_execution_contract_conflicts_rejected": True,
            "registered_target_selection_identity": (
                "spec_id+engine_commit+plugin_commit+config_type"
            ),
            "vision_input_contract_partitions_selection_identity": True,
            "formal_admission_gate_unchanged": True,
        },
        "rejected": rejected,
        "superseded": superseded,
        "satisfied_experiments": satisfied_experiments,
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
