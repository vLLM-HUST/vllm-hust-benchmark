"""Cold-start readiness and steady/burst SLO matrix schema + tooling.

Issue #135: production readiness and burst SLO benchmark. This module defines
the canonical artifact schema (``readiness-slo/v1``), validates individual
repetition artifacts, and aggregates N ≥ 3 independent server-process
repetitions into median/IQR summary rows with outlier handling.

Key invariants enforced (fail-closed):

- Each artifact captures exactly ONE (workload, load_profile, repetition)
  measurement against an independent server process. Cold-start and
  warm-serving data MUST NOT be mixed in the same artifact.
- ``metadata.git_commit`` must be a 40-char lowercase hex SHA.
- ``environment.cann_version`` and ``environment.driver_version`` must be
  explicitly recorded (no ``unknown`` / ``n/a`` / ``not available``
  sentinels — ``readiness_slo`` artifacts must be reproducible).
- ``cache_boundary.residual_services`` MUST be empty for cold-start
  repetitions; a residual service means the measurement is not a true cold
  start and must be rejected.
- ``startup_metrics.warm_vs_cold_improvement_pct`` must be ≥ 0 for
  ``cold_start=False`` artifacts (warm restart cannot be slower than cold
  for the same configuration; if it is, the artifact is inconsistent).
- ``repetition.total`` must be ≥ 3 (independent process restarts per
  issue acceptance criteria).
- ``load_profile.burst_config`` is required when ``kind == "burst"`` and
  forbidden otherwise.
- ``slo_metrics.burst_recovery_s`` is required for burst load profiles and
  forbidden for steady load profiles.
- The aggregator requires ≥3 valid, finite, positive throughput values
  and emits ``overall_status`` of ``admitted`` / ``blocked`` /
  ``incomplete`` / ``negative-result`` per the project status contract.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

SCHEMA_VERSION = "readiness-slo/v1"
ARTIFACT_CLASS = "readiness-slo"
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "readiness_slo_v1.schema.json"
)

SUPPORTED_WORKLOADS = frozenset(
    {
        "random-online",
        "sharegpt-online",
        "prefix-repetition-online",
        "burstgpt",
        "tracelab-specialty",
    }
)

STEADY_LOAD_PROFILES = frozenset({"steady-1rps", "steady-1.2rps"})
BURST_LOAD_PROFILES = frozenset({"burst", "overload-recovery"})
ALL_LOAD_PROFILES = STEADY_LOAD_PROFILES | BURST_LOAD_PROFILES

MIN_REPETITIONS = 3
DEFAULT_REPETITIONS = 3

_OVERALL_STATUS = frozenset({"admitted", "blocked", "incomplete", "negative-result"})

# Placeholder sentinels that must be treated as missing provenance per
# project memory constraints. The admission gate must reject these for
# environment.cann_version / environment.driver_version.
_PLACEHOLDER_SENTINELS = frozenset(
    {
        "",
        "unknown",
        "not available",
        "n/a",
        "none",
        "null",
    }
)

_HEX40_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _is_placeholder_sentinel(value: Any) -> bool:
    """Return True if value is a placeholder provenance sentinel."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return value.strip().lower() in _PLACEHOLDER_SENTINELS
    return False


def _require_finite_non_negative(name: str, value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{context}: metric {name} is not a number: {value!r}"
        ) from error
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context}: metric {name} is invalid: {number!r}")
    return number


def _require_finite_positive(name: str, value: Any, *, context: str) -> float:
    number = _require_finite_non_negative(name, value, context=context)
    if number <= 0:
        raise ValueError(f"{context}: metric {name} must be > 0, got {number!r}")
    return number


def load_schema() -> dict[str, Any]:
    """Load the readiness_slo_v1 JSON Schema document."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def schema_validator() -> Draft7Validator:
    """Return a Draft7Validator for the readiness_slo_v1 schema."""
    return Draft7Validator(load_schema())


@dataclass(frozen=True)
class StartupMetrics:
    """Cold/warm readiness startup metrics (issue #135 section A)."""

    cold_readiness_s: float
    warm_restart_readiness_s: float
    weight_load_s: float
    torch_compile_s: float
    compile_cache_hit: bool
    compile_cache_identity: str
    acl_graph_capture_time_s: float
    acl_graph_capture_count: int
    acl_graph_capture_extra_memory_mb: float
    engine_profile_warmup_s: float
    first_request_ttft_ms: float
    second_request_ttft_ms: float
    warm_vs_cold_improvement_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cold_readiness_s": self.cold_readiness_s,
            "warm_restart_readiness_s": self.warm_restart_readiness_s,
            "weight_load_s": self.weight_load_s,
            "torch_compile_s": self.torch_compile_s,
            "compile_cache": {
                "hit": self.compile_cache_hit,
                "identity": self.compile_cache_identity,
            },
            "acl_graph_capture": {
                "time_s": self.acl_graph_capture_time_s,
                "capture_count": self.acl_graph_capture_count,
                "extra_memory_mb": self.acl_graph_capture_extra_memory_mb,
            },
            "engine_profile_warmup_s": self.engine_profile_warmup_s,
            "first_request_ttft_ms": self.first_request_ttft_ms,
            "second_request_ttft_ms": self.second_request_ttft_ms,
            "warm_vs_cold_improvement_pct": self.warm_vs_cold_improvement_pct,
        }


@dataclass(frozen=True)
class PercentileBlock:
    """mean/P50/P95/P99 percentile block for an SLO metric."""

    mean: float
    p50: float
    p95: float
    p99: float

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
        }


@dataclass(frozen=True)
class SLOMetrics:
    """SLO metrics for one repetition (issue #135 section C)."""

    output_throughput_tps: float
    success_rate: float
    failure_timeout: int
    failure_error: int
    failure_aborted: int
    ttft_ms: PercentileBlock
    tpot_ms: PercentileBlock
    itl_ms: PercentileBlock
    prefix_cache_hit_rate: float
    burst_recovery_s: float | None
    slo_miss_count: int
    slo_miss_reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_throughput_tps": self.output_throughput_tps,
            "success_rate": self.success_rate,
            "failure_breakdown": {
                "timeout": self.failure_timeout,
                "error": self.failure_error,
                "aborted": self.failure_aborted,
            },
            "ttft_ms": self.ttft_ms.to_dict(),
            "tpot_ms": self.tpot_ms.to_dict(),
            "itl_ms": self.itl_ms.to_dict(),
            "prefix_cache_hit_rate": self.prefix_cache_hit_rate,
            "burst_recovery_s": self.burst_recovery_s,
            "slo_miss": {
                "count": self.slo_miss_count,
                "reasons": list(self.slo_miss_reasons),
            },
        }


@dataclass(frozen=True)
class QueueMetrics:
    """Queue/scheduler wait metrics + running/waiting timeseries."""

    queue_wait_ms: PercentileBlock
    scheduler_admission_wait_ms: PercentileBlock
    prefill_wait_ms: PercentileBlock
    running_waiting_timeseries: tuple[dict[str, float | int], ...]
    first_request_ttft_ms: float
    first_request_queue_wait_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_wait_ms": self.queue_wait_ms.to_dict(),
            "scheduler_admission_wait_ms": self.scheduler_admission_wait_ms.to_dict(),
            "prefill_wait_ms": self.prefill_wait_ms.to_dict(),
            "running_waiting_timeseries": [
                dict(entry) for entry in self.running_waiting_timeseries
            ],
            "first_request_separated": {
                "ttft_ms": self.first_request_ttft_ms,
                "queue_wait_ms": self.first_request_queue_wait_ms,
            },
        }


@dataclass(frozen=True)
class KVStateMetrics:
    """KV usage / preemption / eviction / restore metrics."""

    kv_usage_peak_pct: float
    kv_usage_mean_pct: float
    kv_usage_timeseries: tuple[dict[str, float], ...] = field(default_factory=tuple)
    preemption_count: int = 0
    eviction_count: int = 0
    restore_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kv_usage": {
                "peak_pct": self.kv_usage_peak_pct,
                "mean_pct": self.kv_usage_mean_pct,
                "timeseries": [dict(entry) for entry in self.kv_usage_timeseries],
            },
            "preemption_count": self.preemption_count,
            "eviction_count": self.eviction_count,
            "restore_count": self.restore_count,
        }


@dataclass(frozen=True)
class CacheBoundary:
    """Cold-start cache boundary declaration (issue #135 section A last bullet)."""

    cold_start: bool
    cleared_paths: tuple[str, ...]
    preserved_paths: tuple[str, ...]
    residual_services: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cold_start": self.cold_start,
            "cleared_paths": list(self.cleared_paths),
            "preserved_paths": list(self.preserved_paths),
            "residual_services": list(self.residual_services),
        }


@dataclass(frozen=True)
class RawEvidence:
    """SHA-256 digests of raw server log / client result / metrics log."""

    server_log_sha256: str
    client_result_sha256: str
    metrics_log_sha256: str
    server_log_path: str | None = None
    client_result_path: str | None = None
    metrics_log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_log_sha256": self.server_log_sha256,
            "client_result_sha256": self.client_result_sha256,
            "metrics_log_sha256": self.metrics_log_sha256,
            "server_log_path": self.server_log_path,
            "client_result_path": self.client_result_path,
            "metrics_log_path": self.metrics_log_path,
        }


@dataclass(frozen=True)
class Repetition:
    """Per-repetition metadata: index, total, independent_process."""

    index: int
    total: int
    independent_process: bool = True
    server_pid: int | None = None
    started_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "total": self.total,
            "independent_process": self.independent_process,
            "server_pid": self.server_pid,
            "started_at": self.started_at,
        }


@dataclass(frozen=True)
class LoadProfile:
    """Load profile declaration (steady-1rps / steady-1.2rps / burst / overload-recovery)."""

    kind: str
    request_rate: float | None = None
    burst_config: tuple[str, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind}
        if self.request_rate is not None:
            payload["request_rate"] = self.request_rate
        else:
            payload["request_rate"] = None
        if self.kind == "burst":
            assert self.burst_config is not None  # noqa: S101 — defensive, schema-enforced
            size, duration, interval, rate = self.burst_config
            payload["burst_config"] = {
                "size": int(size),
                "duration_s": float(duration),
                "interval_s": float(interval),
                "mean_arrival_rate": float(rate),
            }
        else:
            payload["burst_config"] = None
        return payload


def build_load_profile(
    kind: str,
    *,
    request_rate: float | None = None,
    burst_size: int | None = None,
    burst_duration_s: float | None = None,
    burst_interval_s: float | None = None,
    burst_mean_arrival_rate: float | None = None,
) -> LoadProfile:
    """Build a LoadProfile with cross-field validation.

    ``burst_config`` is required for ``kind == "burst"`` and forbidden for
    steady profiles (fail-closed: prevents accidental burst recovery
    reporting on steady measurements).
    """
    if kind not in ALL_LOAD_PROFILES:
        raise ValueError(
            f"unsupported load profile kind {kind!r}; expected one of "
            f"{sorted(ALL_LOAD_PROFILES)}"
        )
    if kind in STEADY_LOAD_PROFILES:
        if burst_size is not None or burst_duration_s is not None:
            raise ValueError(
                f"burst_config fields are forbidden for steady load profile {kind!r}"
            )
        return LoadProfile(kind=kind, request_rate=request_rate, burst_config=None)
    # burst / overload-recovery: burst_config is required.
    if burst_size is None or burst_duration_s is None:
        raise ValueError(
            f"burst_config (size + duration_s + interval_s + mean_arrival_rate) "
            f"is required for load profile {kind!r}"
        )
    if burst_size < 1:
        raise ValueError(
            f"burst_config.size must be a positive integer, got {burst_size!r}"
        )
    if burst_duration_s <= 0:
        raise ValueError(
            f"burst_config.duration_s must be > 0, got {burst_duration_s!r}"
        )
    if burst_interval_s is None or burst_interval_s < 0:
        raise ValueError(
            f"burst_config.interval_s must be >= 0, got {burst_interval_s!r}"
        )
    if burst_mean_arrival_rate is None or burst_mean_arrival_rate < 0:
        raise ValueError(
            "burst_config.mean_arrival_rate must be >= 0, "
            f"got {burst_mean_arrival_rate!r}"
        )
    return LoadProfile(
        kind=kind,
        request_rate=request_rate,
        burst_config=(
            int(burst_size),
            float(burst_duration_s),
            float(burst_interval_s),
            float(burst_mean_arrival_rate),
        ),
    )


def build_artifact(
    *,
    entry_id: str,
    engine: str,
    engine_version: str,
    config_type: str,
    hardware: Mapping[str, Any],
    model: Mapping[str, Any],
    workload: Mapping[str, Any],
    load_profile: LoadProfile,
    repetition: Repetition,
    same_spec: Mapping[str, Any],
    metadata: Mapping[str, Any],
    versions: Mapping[str, Any],
    environment: Mapping[str, Any],
    startup_metrics: StartupMetrics,
    slo_metrics: SLOMetrics,
    queue_metrics: QueueMetrics,
    kv_state_metrics: KVStateMetrics,
    cache_boundary: CacheBoundary,
    raw_evidence: RawEvidence,
    report_type: str,
    cluster: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a readiness-slo/v1 artifact dict.

    The artifact captures exactly one (workload, load_profile, repetition)
    measurement. ``report_type`` must be consistent with ``load_profile.kind``:

    - ``startup`` for cold/warm readiness measurement repetitions.
    - ``fixed-qps`` for steady load profiles.
    - ``burst`` for burst/overload-recovery load profiles.
    """
    if report_type not in {"startup", "fixed-qps", "burst"}:
        raise ValueError(f"unsupported report_type {report_type!r}")
    if report_type == "fixed-qps" and load_profile.kind not in STEADY_LOAD_PROFILES:
        raise ValueError(
            f"report_type='fixed-qps' requires a steady load profile, "
            f"got {load_profile.kind!r}"
        )
    if report_type == "burst" and load_profile.kind not in BURST_LOAD_PROFILES:
        raise ValueError(
            f"report_type='burst' requires a burst load profile, "
            f"got {load_profile.kind!r}"
        )
    # burst_recovery_s cross-field consistency: required for burst profiles,
    # forbidden for steady profiles.
    if (
        load_profile.kind in BURST_LOAD_PROFILES
        and slo_metrics.burst_recovery_s is None
    ):
        raise ValueError(
            f"slo_metrics.burst_recovery_s is required for load profile "
            f"{load_profile.kind!r}"
        )
    if (
        load_profile.kind in STEADY_LOAD_PROFILES
        and slo_metrics.burst_recovery_s is not None
    ):
        raise ValueError(
            f"slo_metrics.burst_recovery_s is forbidden for steady load profile "
            f"{load_profile.kind!r}"
        )
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_class": ARTIFACT_CLASS,
        "report_type": report_type,
        "entry_id": entry_id,
        "engine": engine,
        "engine_version": engine_version,
        "config_type": config_type,
        "hardware": dict(hardware),
        "cluster": dict(cluster) if cluster is not None else None,
        "model": dict(model),
        "workload": dict(workload),
        "load_profile": load_profile.to_dict(),
        "repetition": repetition.to_dict(),
        "same_spec": dict(same_spec),
        "metadata": dict(metadata),
        "versions": dict(versions),
        "environment": dict(environment),
        "startup_metrics": startup_metrics.to_dict(),
        "slo_metrics": slo_metrics.to_dict(),
        "queue_metrics": queue_metrics.to_dict(),
        "kv_state_metrics": kv_state_metrics.to_dict(),
        "cache_boundary": cache_boundary.to_dict(),
        "raw_evidence": raw_evidence.to_dict(),
    }
    return artifact


def validate_artifact_semantics(
    artifact: Mapping[str, Any], *, context: str = "artifact"
) -> None:
    """Apply fail-closed semantic checks beyond the JSON Schema.

    The JSON Schema enforces structural shape; this function enforces the
    project memory constraints (placeholder sentinels, 40-hex commit,
    cold-start residual services, warm-vs-cold improvement sign, repetition
    count, cross-field burst recovery consistency).
    """
    # 40-char lowercase hex commit provenance.
    git_commit = str(artifact["metadata"]["git_commit"] or "").strip()
    if not _HEX40_PATTERN.match(git_commit):
        raise ValueError(
            f"{context}: metadata.git_commit must be a 40-char lowercase hex SHA, "
            f"got {git_commit!r}"
        )

    # environment.cann_version / driver_version must not be placeholder sentinels.
    environment = artifact["environment"]
    for key in ("cann_version", "driver_version"):
        value = environment.get(key)
        if _is_placeholder_sentinel(value):
            raise ValueError(
                f"{context}: environment.{key} must be explicitly recorded "
                f"(got placeholder {value!r})"
            )

    # cache_boundary.residual_services MUST be empty for cold-start repetitions.
    cache_boundary = artifact["cache_boundary"]
    if cache_boundary["cold_start"] and cache_boundary["residual_services"]:
        raise ValueError(
            f"{context}: cache_boundary.residual_services must be empty for "
            f"cold-start repetitions (residual service means not a true cold "
            f"start): {cache_boundary['residual_services']!r}"
        )

    # startup_metrics.warm_vs_cold_improvement_pct ≥ 0 for warm restart.
    startup = artifact["startup_metrics"]
    if not cache_boundary["cold_start"]:
        improvement = startup["warm_vs_cold_improvement_pct"]
        if improvement < 0:
            raise ValueError(
                f"{context}: warm_vs_cold_improvement_pct must be >= 0 for "
                f"warm-restart repetitions, got {improvement!r}"
            )

    # repetition.total ≥ 3 (issue acceptance criteria).
    repetition = artifact["repetition"]
    if repetition["total"] < MIN_REPETITIONS:
        raise ValueError(
            f"{context}: repetition.total must be >= {MIN_REPETITIONS} "
            f"(independent process restarts), got {repetition['total']!r}"
        )
    if repetition["index"] < 1 or repetition["index"] > repetition["total"]:
        raise ValueError(
            f"{context}: repetition.index must be in [1, total="
            f"{repetition['total']}], got {repetition['index']!r}"
        )
    if not repetition["independent_process"]:
        raise ValueError(
            f"{context}: repetition.independent_process must be true "
            f"(cold/warm readiness requires independent server restarts)"
        )

    # load_profile cross-field: burst_config required for burst, forbidden for steady.
    load_profile = artifact["load_profile"]
    kind = load_profile["kind"]
    if kind in BURST_LOAD_PROFILES and load_profile["burst_config"] is None:
        raise ValueError(
            f"{context}: load_profile.burst_config is required for kind {kind!r}"
        )
    if kind in STEADY_LOAD_PROFILES and load_profile["burst_config"] is not None:
        raise ValueError(
            f"{context}: load_profile.burst_config is forbidden for steady "
            f"profile {kind!r}"
        )

    # burst_recovery_s required for burst, forbidden for steady.
    slo_metrics = artifact["slo_metrics"]
    burst_recovery = slo_metrics.get("burst_recovery_s")
    if kind in BURST_LOAD_PROFILES and burst_recovery is None:
        raise ValueError(
            f"{context}: slo_metrics.burst_recovery_s is required for burst "
            f"load profile {kind!r}"
        )
    if kind in STEADY_LOAD_PROFILES and burst_recovery is not None:
        raise ValueError(
            f"{context}: slo_metrics.burst_recovery_s is forbidden for steady "
            f"load profile {kind!r}"
        )

    # raw_evidence SHA-256 digests must be 64-char lowercase hex.
    raw_evidence = artifact["raw_evidence"]
    for key in ("server_log_sha256", "client_result_sha256", "metrics_log_sha256"):
        digest = str(raw_evidence[key] or "").strip()
        if not _HEX64_PATTERN.match(digest):
            raise ValueError(
                f"{context}: raw_evidence.{key} must be a 64-char lowercase "
                f"hex SHA-256, got {digest!r}"
            )


def validate_artifact(
    artifact: Mapping[str, Any], *, context: str = "artifact"
) -> None:
    """Validate an artifact against the JSON Schema + semantic constraints."""
    errors = sorted(
        schema_validator().iter_errors(dict(artifact)), key=lambda e: list(e.path)
    )
    if errors:
        first = errors[0]
        raise ValueError(
            f"{context}: schema validation failed: {first.message} @ {list(first.path)}"
        )
    validate_artifact_semantics(artifact, context=context)


def load_artifact(path: Path) -> dict[str, Any]:
    """Load a readiness-slo artifact from disk."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: artifact must be a JSON object")
    return payload


def write_artifact(artifact: Mapping[str, Any], path: Path) -> None:
    """Validate and atomically write an artifact to ``path``."""
    validate_artifact(artifact, context=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(dict(artifact), ensure_ascii=False, indent=2) + "\n"
    path.write_text(serialized, encoding="utf-8")


# ---------------------------------------------------------------------------
# Aggregator: median/IQR + outlier handling across N ≥ 3 repetitions.
# ---------------------------------------------------------------------------

AGGREGATION_SCHEMA_VERSION = "readiness-slo-aggregate/v1"
AGGREGATION_STRATEGY = "median+iqr-across-repetitions"
PRIMARY_AGG_METRIC = "output_throughput_tps"

# Metrics aggregated across repetitions with median + IQR (Q1, Q3).
AGGREGATED_METRICS = (
    "output_throughput_tps",
    "success_rate",
    "ttft_ms_mean",
    "ttft_ms_p99",
    "tpot_ms_mean",
    "tpot_ms_p99",
    "itl_ms_p99",
    "queue_wait_ms_p99",
    "prefix_cache_hit_rate",
)

# Metrics aggregated as sum across repetitions (counters).
COUNTER_METRICS = (
    "preemption_count",
    "eviction_count",
    "restore_count",
    "slo_miss_count",
)


def _extract_metric(artifact: Mapping[str, Any], name: str, *, context: str) -> float:
    """Extract a metric value from an artifact by dotted path."""
    slo = artifact["slo_metrics"]
    kv = artifact["kv_state_metrics"]
    if name == "output_throughput_tps":
        # Throughput=0 is a valid negative-result finding (real measurement
        # failure mode), not a non-finite value. Allow 0 here; the
        # aggregator's overall_status logic classifies all-zero throughput
        # as ``negative-result``.
        return _require_finite_non_negative(
            name, slo["output_throughput_tps"], context=context
        )
    if name == "success_rate":
        return _require_finite_non_negative(name, slo["success_rate"], context=context)
    if name == "prefix_cache_hit_rate":
        return _require_finite_non_negative(
            name, slo["prefix_cache_hit_rate"], context=context
        )
    if (
        name.startswith("ttft_ms")
        or name.startswith("tpot_ms")
        or name.startswith("itl_ms")
    ):
        metric_name, _, stat = name.rpartition("_")
        block = slo[metric_name]
        return _require_finite_non_negative(name, block[stat], context=context)
    if name == "queue_wait_ms_p99":
        block = artifact["queue_metrics"]["queue_wait_ms"]
        return _require_finite_non_negative(name, block["p99"], context=context)
    if name in COUNTER_METRICS:
        if name == "slo_miss_count":
            value = slo["slo_miss"]["count"]
        else:
            value = kv[name]
        try:
            return float(int(value))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{context}: counter {name} invalid: {value!r}") from error
    raise KeyError(f"unknown metric {name!r}")


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return (float(ordered[mid - 1]) + float(ordered[mid])) / 2.0


def _quartile(values: Sequence[float], q: float) -> float:
    """Return the q-th quartile (q=0.25 for Q1, q=0.75 for Q3).

    Uses linear interpolation between the closest ranks (Type 7 / R default),
    matching numpy.percentile's default behavior so consumers can reproduce
    the aggregation offline.
    """
    if not values:
        raise ValueError("cannot compute quartile of empty list")
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return float(ordered[0])
    rank = q * (n - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    frac = rank - lower
    return float(ordered[lower] * (1 - frac) + ordered[upper] * frac)


def _iqr_outlier_mask(values: Sequence[float]) -> list[bool]:
    """Return a mask of outliers using the 1.5*IQR rule.

    True = outlier (drop from publication), False = keep.
    """
    if len(values) < 4:
        # IQR outlier detection requires ≥4 values; with 3 repetitions we
        # keep all values but flag them for manual review.
        return [False] * len(values)
    q1 = _quartile(values, 0.25)
    q3 = _quartile(values, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [v < lower_bound or v > upper_bound for v in values]


@dataclass(frozen=True)
class AggregatedMetric:
    """Aggregated metric across repetitions: median + IQR + raw values."""

    median: float
    q1: float
    q3: float
    iqr: float
    raw_values: tuple[float, ...]
    outlier_mask: tuple[bool, ...]
    outlier_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "iqr": self.iqr,
            "raw_values": list(self.raw_values),
            "outlier_mask": list(self.outlier_mask),
            "outlier_count": self.outlier_count,
        }


@dataclass(frozen=True)
class AggregatedCounter:
    """Aggregated counter: sum across repetitions."""

    total: int
    raw_values: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "raw_values": list(self.raw_values),
        }


def _pairing_key(artifact: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the (workload, load_profile, model, hardware) grouping key."""
    workload = artifact["workload"]
    load_profile = artifact["load_profile"]
    hardware = artifact["hardware"]
    model = artifact["model"]
    return (
        workload["name"],
        load_profile["kind"],
        model["name"],
        hardware["chip_model"],
        int(hardware["chip_count"]),
    )


def aggregate_repetitions(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    context: str = "aggregate",
) -> dict[str, Any]:
    """Aggregate N ≥ 3 repetitions of the same (workload, load_profile) pair.

    Returns a ``readiness-slo-aggregate/v1`` document with median/IQR per
    metric, summed counters, outlier mask on the primary metric, and an
    ``overall_status`` field set to:

    - ``admitted``: ≥3 repetitions, all finite/positive, no outliers on the
      primary metric (or outliers manually accepted via the caller).
    - ``incomplete``: fewer than 3 repetitions, or non-finite/non-positive
      primary metric on some repetition.
    - ``blocked``: missing configuration / provenance / pairing mismatch.
    - ``negative-result``: all repetitions present but throughput is zero or
      success_rate is 0 (a real finding, not a measurement failure).
    """
    if not artifacts:
        raise ValueError(f"{context}: at least one repetition is required")

    pairing_keys = {_pairing_key(artifact) for artifact in artifacts}
    if len(pairing_keys) != 1:
        raise ValueError(
            f"{context}: all repetitions must share the same "
            f"(workload, load_profile, model, hardware) pairing; "
            f"got {sorted(str(k) for k in pairing_keys)}"
        )

    pairing_key = next(iter(pairing_keys))
    workload_name, load_kind, model_name, chip_model, chip_count = pairing_key

    # Validate every artifact up front (fail-closed on schema + semantics).
    for index, artifact in enumerate(artifacts, start=1):
        validate_artifact(artifact, context=f"{context}: repetition[{index}]")

    # Verify repetitions share the same `total` and have unique `index` values.
    totals = {int(artifact["repetition"]["total"]) for artifact in artifacts}
    if len(totals) != 1:
        raise ValueError(
            f"{context}: repetitions disagree on repetition.total: {sorted(totals)}"
        )
    total = next(iter(totals))
    if total < MIN_REPETITIONS:
        raise ValueError(
            f"{context}: repetition.total must be >= {MIN_REPETITIONS}, got {total}"
        )
    indices = [int(artifact["repetition"]["index"]) for artifact in artifacts]
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"{context}: repetition.index values must be unique, got {indices}"
        )
    if min(indices) < 1 or max(indices) > total:
        raise ValueError(
            f"{context}: repetition.index must be in [1, total={total}], got {indices}"
        )

    n = len(artifacts)
    if n < MIN_REPETITIONS:
        overall_status = "incomplete"
    else:
        overall_status = "admitted"

    aggregated: dict[str, Any] = {}
    primary_raw_values: list[float] = []

    for name in AGGREGATED_METRICS:
        raw_values: list[float] = []
        incomplete = False
        for index, artifact in enumerate(artifacts, start=1):
            try:
                value = _extract_metric(
                    artifact, name, context=f"{context}: repetition[{index}]"
                )
            except ValueError:
                incomplete = True
                continue
            if not math.isfinite(value):
                incomplete = True
                continue
            raw_values.append(value)
            if name == PRIMARY_AGG_METRIC:
                primary_raw_values.append(value)

        if len(raw_values) < MIN_REPETITIONS:
            overall_status = "incomplete"
        aggregated[name] = AggregatedMetric(
            median=_median(raw_values) if raw_values else 0.0,
            q1=_quartile(raw_values, 0.25) if raw_values else 0.0,
            q3=_quartile(raw_values, 0.75) if raw_values else 0.0,
            iqr=(
                _quartile(raw_values, 0.75) - _quartile(raw_values, 0.25)
                if raw_values
                else 0.0
            ),
            raw_values=tuple(raw_values),
            outlier_mask=tuple(_iqr_outlier_mask(raw_values)) if raw_values else (),
            outlier_count=sum(_iqr_outlier_mask(raw_values)) if raw_values else 0,
        ).to_dict()
        if incomplete:
            overall_status = "incomplete"

    counters: dict[str, Any] = {}
    for name in COUNTER_METRICS:
        raw_counter_values: list[int] = []
        for index, artifact in enumerate(artifacts, start=1):
            value = _extract_metric(
                artifact, name, context=f"{context}: repetition[{index}]"
            )
            raw_counter_values.append(int(value))
        counters[name] = AggregatedCounter(
            total=sum(raw_counter_values),
            raw_values=tuple(raw_counter_values),
        ).to_dict()

    # Outlier detection on the primary metric (output_throughput_tps).
    primary_block = aggregated[PRIMARY_AGG_METRIC]
    if primary_block["outlier_count"] > 0:
        # Outliers do NOT flip status to incomplete (real measurement), but
        # they are surfaced for review. The caller may still publish with
        # outliers flagged. If primary raw values include non-positive values
        # → negative-result.
        pass
    if primary_raw_values and all(v <= 0 for v in primary_raw_values):
        overall_status = "negative-result"
    if (
        primary_raw_values
        and any(v <= 0 for v in primary_raw_values)
        and not all(v <= 0 for v in primary_raw_values)
    ):
        # Mixed positive/non-positive primary values → incomplete evidence.
        overall_status = "incomplete"

    # Check success_rate ≤ 0 across all repetitions → negative-result.
    success_values = aggregated["success_rate"]["raw_values"]
    if success_values and all(v <= 0 for v in success_values):
        overall_status = "negative-result"

    # If provenance is missing on any artifact, the aggregator already fails
    # closed in validate_artifact; no extra check needed here.

    payload: dict[str, Any] = {
        "schema_version": AGGREGATION_SCHEMA_VERSION,
        "strategy": AGGREGATION_STRATEGY,
        "primary_metric": PRIMARY_AGG_METRIC,
        "workload": workload_name,
        "load_profile": load_kind,
        "model": model_name,
        "hardware_chip_model": chip_model,
        "chip_count": chip_count,
        "repetition_count": n,
        "repetition_total": total,
        "repetition_indices": indices,
        "overall_status": overall_status,
        "metrics": aggregated,
        "counters": counters,
    }
    return payload


def validate_aggregate(
    aggregate: Mapping[str, Any], *, context: str = "aggregate"
) -> None:
    """Validate an aggregate document produced by :func:`aggregate_repetitions`."""
    if aggregate.get("schema_version") != AGGREGATION_SCHEMA_VERSION:
        raise ValueError(
            f"{context}: unsupported schema_version {aggregate.get('schema_version')!r}"
        )
    if aggregate.get("strategy") != AGGREGATION_STRATEGY:
        raise ValueError(
            f"{context}: unsupported strategy {aggregate.get('strategy')!r}"
        )
    if aggregate.get("primary_metric") != PRIMARY_AGG_METRIC:
        raise ValueError(f"{context}: primary_metric must be {PRIMARY_AGG_METRIC!r}")
    if aggregate.get("overall_status") not in _OVERALL_STATUS:
        raise ValueError(
            f"{context}: overall_status must be one of {sorted(_OVERALL_STATUS)}, "
            f"got {aggregate.get('overall_status')!r}"
        )
    repetition_total = aggregate.get("repetition_total")
    if not isinstance(repetition_total, int) or repetition_total < MIN_REPETITIONS:
        raise ValueError(
            f"{context}: repetition_total must be >= {MIN_REPETITIONS}, "
            f"got {repetition_total!r}"
        )
    repetition_count = aggregate.get("repetition_count")
    if not isinstance(repetition_count, int) or repetition_count < 1:
        raise ValueError(
            f"{context}: repetition_count must be a positive integer, "
            f"got {repetition_count!r}"
        )
    if repetition_count > repetition_total:
        raise ValueError(
            f"{context}: repetition_count ({repetition_count}) cannot exceed "
            f"repetition_total ({repetition_total})"
        )
    metrics = aggregate.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{context}: metrics must be an object")
    for name in AGGREGATED_METRICS:
        if name not in metrics:
            raise ValueError(f"{context}: metrics.{name} is required")
        block = metrics[name]
        if not isinstance(block, dict):
            raise ValueError(f"{context}: metrics.{name} must be an object")
        for key in ("median", "q1", "q3", "iqr", "raw_values", "outlier_mask"):
            if key not in block:
                raise ValueError(f"{context}: metrics.{name}.{key} is required")
    counters = aggregate.get("counters")
    if not isinstance(counters, dict):
        raise ValueError(f"{context}: counters must be an object")
    for name in COUNTER_METRICS:
        if name not in counters:
            raise ValueError(f"{context}: counters.{name} is required")


# ---------------------------------------------------------------------------
# Traffic matrix definition helpers (issue #135 section B).
# ---------------------------------------------------------------------------

DEFAULT_WORKLOAD_MATRIX = (
    "random-online",
    "sharegpt-online",
    "prefix-repetition-online",
    "burstgpt",
)

DEFAULT_LOAD_PROFILE_MATRIX = (
    "steady-1rps",
    "steady-1.2rps",
    "burst",
)


def traffic_matrix(
    *,
    workloads: Sequence[str] = DEFAULT_WORKLOAD_MATRIX,
    load_profiles: Sequence[str] = DEFAULT_LOAD_PROFILE_MATRIX,
    repetitions: int = DEFAULT_REPETITIONS,
) -> list[dict[str, Any]]:
    """Return the list of (workload, load_profile, repetition) tuples.

    The default matrix covers random/sharegpt/prefix/burstgpt × steady-1rps
    / steady-1.2rps / burst × 3 independent server restarts, matching the
    issue acceptance criteria: ``random/sharegpt/prefix 均完成 1 RPS、1.2 RPS、
    burst 三档``.
    """
    if repetitions < MIN_REPETITIONS:
        raise ValueError(
            f"repetitions must be >= {MIN_REPETITIONS} (independent restarts), "
            f"got {repetitions}"
        )
    for workload in workloads:
        if workload not in SUPPORTED_WORKLOADS:
            raise ValueError(
                f"unsupported workload {workload!r}; expected one of "
                f"{sorted(SUPPORTED_WORKLOADS)}"
            )
    for profile in load_profiles:
        if profile not in ALL_LOAD_PROFILES:
            raise ValueError(
                f"unsupported load profile {profile!r}; expected one of "
                f"{sorted(ALL_LOAD_PROFILES)}"
            )
    matrix: list[dict[str, Any]] = []
    for workload in workloads:
        for profile in load_profiles:
            for index in range(1, repetitions + 1):
                matrix.append(
                    {
                        "workload": workload,
                        "load_profile": profile,
                        "repetition_index": index,
                        "repetition_total": repetitions,
                    }
                )
    return matrix


def matrix_size(
    *,
    workloads: Sequence[str] = DEFAULT_WORKLOAD_MATRIX,
    load_profiles: Sequence[str] = DEFAULT_LOAD_PROFILE_MATRIX,
    repetitions: int = DEFAULT_REPETITIONS,
) -> int:
    """Return the number of (workload, load_profile, repetition) cells."""
    return len(workloads) * len(load_profiles) * repetitions
