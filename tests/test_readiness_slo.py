"""Tests for readiness_slo schema, builder, validator, and aggregator.

Issue #135: cold-start readiness + steady/1.2RPS/burst SLO matrix.

Covers:
- JSON Schema structural validation (accept valid; reject missing fields,
  wrong types, bad enum/pattern values).
- Semantic validation (fail-closed for placeholder sentinels, cold-start
  residual services, warm-vs-cold improvement sign, repetition count,
  cross-field burst_config/burst_recovery_s consistency, 40-hex commit,
  64-hex evidence digests).
- Builder functions (build_load_profile, build_artifact) with cross-field
  validation.
- Aggregator: median/IQR + outlier mask; ≥3 repetitions → admitted;
  <3 → incomplete; all-zero throughput → negative-result; mixed
  positive/non-positive → incomplete; pairing mismatch → blocked.
- Traffic matrix helper: default size, custom workloads, repetitions < 3
  rejected, bad workloads/profiles rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_hust_benchmark import readiness_slo
from vllm_hust_benchmark.readiness_slo import (
    AGGREGATED_METRICS,
    AGGREGATION_SCHEMA_VERSION,
    AGGREGATION_STRATEGY,
    ALL_LOAD_PROFILES,
    ARTIFACT_CLASS,
    COUNTER_METRICS,
    DEFAULT_LOAD_PROFILE_MATRIX,
    DEFAULT_WORKLOAD_MATRIX,
    MIN_REPETITIONS,
    PRIMARY_AGG_METRIC,
    SCHEMA_VERSION,
    SUPPORTED_WORKLOADS,
    CacheBoundary,
    KVStateMetrics,
    PercentileBlock,
    QueueMetrics,
    RawEvidence,
    Repetition,
    SLOMetrics,
    StartupMetrics,
    aggregate_repetitions,
    build_artifact,
    build_load_profile,
    load_artifact,
    matrix_size,
    schema_validator,
    traffic_matrix,
    validate_aggregate,
    validate_artifact_semantics,
    write_artifact,
)

COMMIT_40 = "a" * 40
SHA256_64 = "b" * 64


def _percentile(value: float = 100.0) -> PercentileBlock:
    return PercentileBlock(mean=value, p50=value, p95=value * 1.1, p99=value * 1.2)


def _startup_metrics(cold: bool = True) -> StartupMetrics:
    return StartupMetrics(
        cold_readiness_s=120.0 if cold else 30.0,
        warm_restart_readiness_s=30.0,
        weight_load_s=20.0,
        torch_compile_s=15.0,
        compile_cache_hit=not cold,
        compile_cache_identity="sha256:" + "c" * 64,
        acl_graph_capture_time_s=10.0,
        acl_graph_capture_count=4,
        acl_graph_capture_extra_memory_mb=512.0,
        engine_profile_warmup_s=5.0,
        first_request_ttft_ms=200.0,
        second_request_ttft_ms=30.0,
        warm_vs_cold_improvement_pct=75.0 if not cold else 0.0,
    )


def _slo_metrics(burst: bool = False) -> SLOMetrics:
    return SLOMetrics(
        output_throughput_tps=42.5,
        success_rate=1.0,
        failure_timeout=0,
        failure_error=0,
        failure_aborted=0,
        ttft_ms=_percentile(50.0),
        tpot_ms=_percentile(20.0),
        itl_ms=_percentile(15.0),
        prefix_cache_hit_rate=0.6,
        burst_recovery_s=12.5 if burst else None,
        slo_miss_count=0,
        slo_miss_reasons=(),
    )


def _queue_metrics() -> QueueMetrics:
    return QueueMetrics(
        queue_wait_ms=_percentile(40.0),
        scheduler_admission_wait_ms=_percentile(5.0),
        prefill_wait_ms=_percentile(80.0),
        running_waiting_timeseries=(
            {"t": 0.0, "running": 1, "waiting": 0},
            {"t": 1.0, "running": 2, "waiting": 1},
        ),
        first_request_ttft_ms=200.0,
        first_request_queue_wait_ms=30.0,
    )


def _kv_state_metrics() -> KVStateMetrics:
    return KVStateMetrics(
        kv_usage_peak_pct=85.0,
        kv_usage_mean_pct=60.0,
        kv_usage_timeseries=({"t": 0.0, "pct": 50.0},),
        preemption_count=0,
        eviction_count=0,
        restore_count=0,
    )


def _cache_boundary(cold: bool = True) -> CacheBoundary:
    return CacheBoundary(
        cold_start=cold,
        cleared_paths=("/tmp/vllm-cache",) if cold else (),
        preserved_paths=() if cold else ("/tmp/vllm-cache",),
        residual_services=(),
    )


def _raw_evidence() -> RawEvidence:
    return RawEvidence(
        server_log_sha256=SHA256_64,
        client_result_sha256=SHA256_64,
        metrics_log_sha256=SHA256_64,
        server_log_path="/tmp/server.log",
        client_result_path="/tmp/client.json",
        metrics_log_path="/tmp/metrics.log",
    )


def _repetition(index: int = 1, total: int = 3) -> Repetition:
    return Repetition(index=index, total=total, independent_process=True)


def _build_artifact(
    *,
    cold: bool = True,
    profile: str | None = None,
    rep_index: int = 1,
    rep_total: int = 3,
    burst: bool = False,
    commit: str = COMMIT_40,
    cann_version: str = "8.0.0",
    driver_version: str = "23.0.0",
    throughput: float = 42.5,
    success_rate: float = 1.0,
    residual_services: tuple[str, ...] = (),
    warm_improvement: float = 75.0,
    report_type: str | None = None,
) -> dict:
    # Derive profile from `burst` flag when not explicitly passed.
    if profile is None:
        profile = "burst" if burst else "steady-1rps"
    if report_type is None:
        report_type = "burst" if burst else "fixed-qps"

    # Build the load profile via the canonical builder so the artifact's
    # burst_config is consistent with its kind.
    if burst:
        load_profile = build_load_profile(
            profile,
            request_rate=None,
            burst_size=50,
            burst_duration_s=30.0,
            burst_interval_s=10.0,
            burst_mean_arrival_rate=5.0,
        )
    else:
        load_profile = build_load_profile(profile, request_rate=1.0)

    startup = _startup_metrics(cold=cold)
    startup = StartupMetrics(
        **{**startup.__dict__, "warm_vs_cold_improvement_pct": warm_improvement}
    )

    slo = _slo_metrics(burst=burst)
    slo = SLOMetrics(
        **{
            **slo.__dict__,
            "output_throughput_tps": throughput,
            "success_rate": success_rate,
        }
    )

    cache = _cache_boundary(cold=cold)
    cache = CacheBoundary(
        cold_start=cold,
        cleared_paths=cache.cleared_paths,
        preserved_paths=cache.preserved_paths,
        residual_services=residual_services,
    )

    return build_artifact(
        entry_id=f"test-{profile}-rep{rep_index}",
        engine="vllm-hust",
        engine_version="v0.18.0",
        config_type="single_gpu",
        hardware={
            "vendor": "Huawei",
            "chip_model": "910B3",
            "chip_count": 1,
            "interconnect": "unknown",
        },
        model={
            "name": "Qwen/Qwen2.5-14B-Instruct",
            "parameters": "14B",
            "precision": "FP16",
            "quantization": None,
            "canonical_id": "hf:Qwen/Qwen2.5-14B-Instruct",
            "short_name": "Qwen2.5-14B-Instruct",
            "display_name": "Qwen2.5-14B-Instruct",
        },
        workload={
            "name": "random-online",
            "dataset": "random",
            "input_length": 1024,
            "output_length": 256,
            "batch_size": None,
            "concurrent_requests": None,
        },
        load_profile=load_profile,
        repetition=_repetition(index=rep_index, total=rep_total),
        same_spec={
            "spec_id": "issue-135-readiness-slo-random-online-steady-1rps",
            "spec_label": "Issue #135 readiness SLO matrix",
            "scenario": "random-online",
            "resolved_spec_hash": None,
            "resolved_server_parameters": {"gpu_memory_utilization": 0.6},
            "resolved_client_parameters": {"request_rate": 1},
        },
        metadata={
            "submitted_at": "2026-08-06T10:00:00Z",
            "submitter": "issue-135-tests",
            "data_source": "issue-135-readiness-slo-matrix",
            "engine": "vllm-hust",
            "engine_version": "v0.18.0",
            "git_commit": commit,
            "github_repository": "vLLM-HUST/vllm-hust",
            "github_ref": "main",
            "verified": True,
            "idempotency_key": f"{commit}-rep{rep_index}",
        },
        versions={
            "protocol": "N/A",
            "backend": "0.1.0",
            "core": "v0.18.0",
            "benchmark": "0.1.0",
        },
        environment={
            "os": "Linux-5.10.0-aarch64",
            "python_version": "3.11.15",
            "pytorch_version": "2.1.0",
            "cuda_version": None,
            "cann_version": cann_version,
            "driver_version": driver_version,
        },
        startup_metrics=startup,
        slo_metrics=slo,
        queue_metrics=_queue_metrics(),
        kv_state_metrics=_kv_state_metrics(),
        cache_boundary=cache,
        raw_evidence=_raw_evidence(),
        report_type=report_type,
    )


# ---------------------------------------------------------------------------
# Schema-level tests.
# ---------------------------------------------------------------------------


def test_schema_loads_and_validator_constructs() -> None:
    schema = readiness_slo.load_schema()
    assert schema["title"] == "Readiness SLO v1"
    validator = schema_validator()
    assert validator.schema["title"] == "Readiness SLO v1"


def test_schema_constants_match() -> None:
    assert SCHEMA_VERSION == "readiness-slo/v1"
    assert ARTIFACT_CLASS == "readiness-slo"
    assert MIN_REPETITIONS == 3
    assert SUPPORTED_WORKLOADS == {
        "random-online",
        "sharegpt-online",
        "prefix-repetition-online",
        "burstgpt",
        "tracelab-specialty",
    }
    assert ALL_LOAD_PROFILES == {
        "steady-1rps",
        "steady-1.2rps",
        "burst",
        "overload-recovery",
    }


def test_valid_artifact_passes_schema_validation() -> None:
    artifact = _build_artifact()
    errors = sorted(
        schema_validator().iter_errors(artifact), key=lambda e: list(e.path)
    )
    assert errors == []


def test_missing_required_field_fails_schema() -> None:
    artifact = _build_artifact()
    del artifact["startup_metrics"]
    errors = sorted(
        schema_validator().iter_errors(artifact), key=lambda e: list(e.path)
    )
    assert errors, "expected schema validation to fail when startup_metrics is missing"


def test_bad_commit_pattern_fails_schema() -> None:
    artifact = _build_artifact(commit="short")
    errors = sorted(
        schema_validator().iter_errors(artifact), key=lambda e: list(e.path)
    )
    assert errors, "expected schema validation to fail for non-40-hex commit"


def test_bad_workload_enum_fails_schema() -> None:
    artifact = _build_artifact()
    artifact["workload"]["name"] = "unknown-workload"
    errors = sorted(
        schema_validator().iter_errors(artifact), key=lambda e: list(e.path)
    )
    assert errors


def test_bad_load_profile_kind_fails_schema() -> None:
    artifact = _build_artifact()
    artifact["load_profile"]["kind"] = "steady-2rps"
    errors = sorted(
        schema_validator().iter_errors(artifact), key=lambda e: list(e.path)
    )
    assert errors


def test_extra_property_fails_schema() -> None:
    artifact = _build_artifact()
    artifact["unexpected_field"] = "value"
    errors = sorted(
        schema_validator().iter_errors(artifact), key=lambda e: list(e.path)
    )
    assert errors


def test_burst_recovery_outside_range_fails_schema() -> None:
    artifact = _build_artifact(burst=True)
    artifact["slo_metrics"]["burst_recovery_s"] = -1.0
    errors = sorted(
        schema_validator().iter_errors(artifact), key=lambda e: list(e.path)
    )
    assert errors


# ---------------------------------------------------------------------------
# Semantic validation tests (fail-closed).
# ---------------------------------------------------------------------------


def test_valid_artifact_passes_semantic_validation() -> None:
    artifact = _build_artifact()
    validate_artifact_semantics(artifact)


def test_placeholder_cann_version_rejected() -> None:
    for sentinel in ("unknown", "n/a", "not available", "none", "null", ""):
        artifact = _build_artifact(cann_version=sentinel)
        with pytest.raises(
            ValueError, match="environment.cann_version must be explicitly recorded"
        ):
            validate_artifact_semantics(artifact)


def test_placeholder_driver_version_rejected() -> None:
    for sentinel in ("unknown", "n/a", "not available", "none", "null", ""):
        artifact = _build_artifact(driver_version=sentinel)
        with pytest.raises(
            ValueError, match="environment.driver_version must be explicitly recorded"
        ):
            validate_artifact_semantics(artifact)


def test_cold_start_with_residual_services_rejected() -> None:
    artifact = _build_artifact(cold=True, residual_services=("vllm-server",))
    with pytest.raises(
        ValueError, match="residual_services must be empty for cold-start"
    ):
        validate_artifact_semantics(artifact)


def test_warm_start_with_negative_improvement_rejected() -> None:
    artifact = _build_artifact(cold=False, warm_improvement=-5.0)
    with pytest.raises(ValueError, match="warm_vs_cold_improvement_pct must be >= 0"):
        validate_artifact_semantics(artifact)


def test_repetition_total_below_minimum_rejected() -> None:
    artifact = _build_artifact(rep_total=2)
    with pytest.raises(ValueError, match="repetition.total must be >= 3"):
        validate_artifact_semantics(artifact)


def test_repetition_index_out_of_range_rejected() -> None:
    artifact = _build_artifact(rep_index=4, rep_total=3)
    with pytest.raises(ValueError, match="repetition.index must be in"):
        validate_artifact_semantics(artifact)


def test_non_independent_process_rejected() -> None:
    artifact = _build_artifact()
    artifact["repetition"]["independent_process"] = False
    with pytest.raises(ValueError, match="independent_process must be true"):
        validate_artifact_semantics(artifact)


def test_steady_profile_with_burst_config_rejected() -> None:
    artifact = _build_artifact(profile="steady-1rps")
    artifact["load_profile"]["burst_config"] = {
        "size": 10,
        "duration_s": 5.0,
        "interval_s": 1.0,
        "mean_arrival_rate": 2.0,
    }
    with pytest.raises(ValueError, match="burst_config is forbidden for steady"):
        validate_artifact_semantics(artifact)


def test_burst_profile_without_burst_config_rejected() -> None:
    artifact = _build_artifact(burst=True)
    artifact["load_profile"]["burst_config"] = None
    with pytest.raises(ValueError, match="burst_config is required for kind 'burst'"):
        validate_artifact_semantics(artifact)


def test_steady_profile_with_burst_recovery_rejected() -> None:
    artifact = _build_artifact(profile="steady-1rps")
    artifact["slo_metrics"]["burst_recovery_s"] = 10.0
    with pytest.raises(ValueError, match="burst_recovery_s is forbidden for steady"):
        validate_artifact_semantics(artifact)


def test_burst_profile_without_burst_recovery_rejected() -> None:
    artifact = _build_artifact(burst=True)
    artifact["slo_metrics"]["burst_recovery_s"] = None
    with pytest.raises(ValueError, match="burst_recovery_s is required for burst"):
        validate_artifact_semantics(artifact)


def test_short_commit_rejected() -> None:
    artifact = _build_artifact(commit="abc123")
    with pytest.raises(ValueError, match="40-char lowercase hex SHA"):
        validate_artifact_semantics(artifact)


def test_uppercase_commit_rejected() -> None:
    artifact = _build_artifact(commit="A" * 40)
    with pytest.raises(ValueError, match="40-char lowercase hex SHA"):
        validate_artifact_semantics(artifact)


def test_short_evidence_digest_rejected() -> None:
    artifact = _build_artifact()
    artifact["raw_evidence"]["server_log_sha256"] = "abc"
    with pytest.raises(ValueError, match="server_log_sha256 must be a 64-char"):
        validate_artifact_semantics(artifact)


# ---------------------------------------------------------------------------
# Builder tests.
# ---------------------------------------------------------------------------


def test_build_load_profile_steady() -> None:
    profile = build_load_profile("steady-1rps", request_rate=1.0)
    payload = profile.to_dict()
    assert payload["kind"] == "steady-1rps"
    assert payload["burst_config"] is None
    assert payload["request_rate"] == 1.0


def test_build_load_profile_burst() -> None:
    profile = build_load_profile(
        "burst",
        burst_size=50,
        burst_duration_s=30.0,
        burst_interval_s=10.0,
        burst_mean_arrival_rate=5.0,
    )
    payload = profile.to_dict()
    assert payload["kind"] == "burst"
    assert payload["burst_config"]["size"] == 50
    assert payload["burst_config"]["duration_s"] == 30.0


def test_build_load_profile_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unsupported load profile kind"):
        build_load_profile("steady-2rps")


def test_build_load_profile_rejects_burst_config_on_steady() -> None:
    with pytest.raises(ValueError, match="burst_config fields are forbidden"):
        build_load_profile("steady-1rps", burst_size=10, burst_duration_s=5.0)


def test_build_load_profile_rejects_burst_without_burst_config() -> None:
    with pytest.raises(ValueError, match="burst_config .* is required"):
        build_load_profile("burst")


def test_build_load_profile_rejects_zero_burst_duration() -> None:
    with pytest.raises(ValueError, match="duration_s must be > 0"):
        build_load_profile(
            "burst",
            burst_size=10,
            burst_duration_s=0.0,
            burst_interval_s=1.0,
            burst_mean_arrival_rate=2.0,
        )


def test_build_artifact_rejects_burst_report_type_on_steady_profile() -> None:
    """Builder rejects report_type='burst' on a steady load profile."""
    with pytest.raises(ValueError, match="report_type='burst' requires a burst"):
        _build_artifact(burst=False, report_type="burst")


def test_build_artifact_rejects_fixed_qps_report_type_on_burst_profile() -> None:
    """Builder rejects report_type='fixed-qps' on a burst load profile."""
    with pytest.raises(ValueError, match="report_type='fixed-qps' requires a steady"):
        _build_artifact(burst=True, report_type="fixed-qps")


def test_validator_rejects_burst_recovery_on_steady() -> None:
    """Semantic validator rejects burst_recovery_s on a steady profile."""
    artifact = _build_artifact(burst=False)
    artifact["slo_metrics"]["burst_recovery_s"] = 10.0
    with pytest.raises(ValueError, match="burst_recovery_s is forbidden for steady"):
        validate_artifact_semantics(artifact)


def test_validator_rejects_missing_burst_recovery_on_burst() -> None:
    """Semantic validator rejects missing burst_recovery_s on a burst profile."""
    artifact = _build_artifact(burst=True)
    artifact["slo_metrics"]["burst_recovery_s"] = None
    with pytest.raises(ValueError, match="burst_recovery_s is required for burst"):
        validate_artifact_semantics(artifact)


# ---------------------------------------------------------------------------
# Aggregator tests.
# ---------------------------------------------------------------------------


def _three_repetitions(
    throughputs: tuple[float, ...] = (40.0, 50.0, 60.0),
) -> list[dict]:
    return [
        _build_artifact(rep_index=i, rep_total=3, throughput=t)
        for i, t in enumerate(throughputs, start=1)
    ]


def test_aggregate_three_repetitions_admitted() -> None:
    artifacts = _three_repetitions((40.0, 50.0, 60.0))
    aggregate = aggregate_repetitions(artifacts)
    assert aggregate["schema_version"] == AGGREGATION_SCHEMA_VERSION
    assert aggregate["strategy"] == AGGREGATION_STRATEGY
    assert aggregate["primary_metric"] == PRIMARY_AGG_METRIC
    assert aggregate["repetition_count"] == 3
    assert aggregate["repetition_total"] == 3
    assert aggregate["overall_status"] == "admitted"
    # Median of (40, 50, 60) is 50.
    assert aggregate["metrics"]["output_throughput_tps"]["median"] == 50.0
    assert aggregate["metrics"]["output_throughput_tps"]["raw_values"] == [
        40.0,
        50.0,
        60.0,
    ]


def test_aggregate_two_repetitions_incomplete() -> None:
    artifacts = [
        _build_artifact(rep_index=1, rep_total=3, throughput=40.0),
        _build_artifact(rep_index=2, rep_total=3, throughput=50.0),
    ]
    aggregate = aggregate_repetitions(artifacts)
    assert aggregate["overall_status"] == "incomplete"


def test_aggregate_all_zero_throughput_negative_result() -> None:
    artifacts = [
        _build_artifact(rep_index=i, rep_total=3, throughput=0.0) for i in range(1, 4)
    ]
    # Schema allows throughput=0 (min: 0); semantic validator allows it too;
    # aggregator must classify as negative-result.
    aggregate = aggregate_repetitions(artifacts)
    assert aggregate["overall_status"] == "negative-result"


def test_aggregate_mixed_positive_and_zero_incomplete() -> None:
    artifacts = [
        _build_artifact(rep_index=1, rep_total=3, throughput=0.0),
        _build_artifact(rep_index=2, rep_total=3, throughput=50.0),
        _build_artifact(rep_index=3, rep_total=3, throughput=60.0),
    ]
    aggregate = aggregate_repetitions(artifacts)
    assert aggregate["overall_status"] == "incomplete"


def test_aggregate_pairing_mismatch_fails() -> None:
    artifacts = [
        _build_artifact(rep_index=1, rep_total=3, throughput=40.0),
        _build_artifact(rep_index=2, rep_total=3, throughput=50.0),
        _build_artifact(rep_index=3, rep_total=3, throughput=60.0),
    ]
    # Mismatch workload name on repetition 3.
    artifacts[2]["workload"]["name"] = "sharegpt-online"
    with pytest.raises(ValueError, match="must share the same"):
        aggregate_repetitions(artifacts)


def test_aggregate_repetition_total_mismatch_fails() -> None:
    artifacts = [
        _build_artifact(rep_index=1, rep_total=3, throughput=40.0),
        _build_artifact(rep_index=2, rep_total=3, throughput=50.0),
        _build_artifact(rep_index=3, rep_total=5, throughput=60.0),
    ]
    with pytest.raises(ValueError, match="disagree on repetition.total"):
        aggregate_repetitions(artifacts)


def test_aggregate_duplicate_indices_fails() -> None:
    artifacts = [
        _build_artifact(rep_index=1, rep_total=3, throughput=40.0),
        _build_artifact(rep_index=1, rep_total=3, throughput=50.0),
        _build_artifact(rep_index=2, rep_total=3, throughput=60.0),
    ]
    with pytest.raises(ValueError, match="repetition.index values must be unique"):
        aggregate_repetitions(artifacts)


def test_aggregate_schema_invalid_artifact_fails() -> None:
    artifacts = _three_repetitions()
    # Inject a placeholder cann_version on repetition 1.
    artifacts[0]["environment"]["cann_version"] = "unknown"
    with pytest.raises(ValueError, match="cann_version must be explicitly recorded"):
        aggregate_repetitions(artifacts)


def test_aggregate_counters_summed_across_repetitions() -> None:
    artifacts = [
        _build_artifact(rep_index=i, rep_total=3, throughput=50.0) for i in range(1, 4)
    ]
    # Inject preemption counts.
    for i, artifact in enumerate(artifacts, start=1):
        artifact["kv_state_metrics"]["preemption_count"] = i
    aggregate = aggregate_repetitions(artifacts)
    assert aggregate["counters"]["preemption_count"]["total"] == 6  # 1+2+3
    assert aggregate["counters"]["preemption_count"]["raw_values"] == [1, 2, 3]


def test_aggregate_median_calculation_with_five_repetitions() -> None:
    throughputs = (10.0, 20.0, 30.0, 40.0, 50.0)
    artifacts = [
        _build_artifact(rep_index=i, rep_total=5, throughput=t)
        for i, t in enumerate(throughputs, start=1)
    ]
    aggregate = aggregate_repetitions(artifacts)
    assert aggregate["repetition_count"] == 5
    assert aggregate["metrics"]["output_throughput_tps"]["median"] == 30.0
    # Q1 = 20, Q3 = 40, IQR = 20 (Type 7 interpolation).
    assert aggregate["metrics"]["output_throughput_tps"]["q1"] == 20.0
    assert aggregate["metrics"]["output_throughput_tps"]["q3"] == 40.0
    assert aggregate["metrics"]["output_throughput_tps"]["iqr"] == 20.0


def test_aggregate_outlier_detection_on_large_set() -> None:
    # 5 values with one obvious outlier (1000).
    throughputs = (40.0, 50.0, 60.0, 70.0, 1000.0)
    artifacts = [
        _build_artifact(rep_index=i, rep_total=5, throughput=t)
        for i, t in enumerate(throughputs, start=1)
    ]
    aggregate = aggregate_repetitions(artifacts)
    mask = aggregate["metrics"]["output_throughput_tps"]["outlier_mask"]
    assert any(mask), "expected the 1000.0 value to be flagged as outlier"
    assert aggregate["metrics"]["output_throughput_tps"]["outlier_count"] == 1


def test_aggregate_outlier_detection_not_run_on_three_values() -> None:
    # With exactly 3 repetitions, IQR outlier detection is not run.
    artifacts = _three_repetitions((40.0, 50.0, 60.0))
    aggregate = aggregate_repetitions(artifacts)
    assert aggregate["metrics"]["output_throughput_tps"]["outlier_mask"] == [
        False,
        False,
        False,
    ]
    assert aggregate["metrics"]["output_throughput_tps"]["outlier_count"] == 0


def test_aggregate_includes_all_required_metrics() -> None:
    artifacts = _three_repetitions()
    aggregate = aggregate_repetitions(artifacts)
    for name in AGGREGATED_METRICS:
        assert name in aggregate["metrics"], f"missing aggregated metric {name}"
    for name in COUNTER_METRICS:
        assert name in aggregate["counters"], f"missing counter {name}"


def test_validate_aggregate_accepts_valid() -> None:
    aggregate = aggregate_repetitions(_three_repetitions())
    validate_aggregate(aggregate)


def test_validate_aggregate_rejects_bad_schema_version() -> None:
    aggregate = aggregate_repetitions(_three_repetitions())
    aggregate["schema_version"] = "wrong/v1"
    with pytest.raises(ValueError, match="unsupported schema_version"):
        validate_aggregate(aggregate)


def test_validate_aggregate_rejects_bad_overall_status() -> None:
    aggregate = aggregate_repetitions(_three_repetitions())
    aggregate["overall_status"] = "garbage"
    with pytest.raises(ValueError, match="overall_status must be one of"):
        validate_aggregate(aggregate)


def test_validate_aggregate_rejects_missing_metric() -> None:
    aggregate = aggregate_repetitions(_three_repetitions())
    del aggregate["metrics"]["output_throughput_tps"]
    with pytest.raises(ValueError, match="metrics.output_throughput_tps is required"):
        validate_aggregate(aggregate)


def test_validate_aggregate_rejects_low_repetition_total() -> None:
    aggregate = aggregate_repetitions(_three_repetitions())
    aggregate["repetition_total"] = 2
    with pytest.raises(ValueError, match="repetition_total must be >= 3"):
        validate_aggregate(aggregate)


# ---------------------------------------------------------------------------
# Traffic matrix tests.
# ---------------------------------------------------------------------------


def test_traffic_matrix_default_size() -> None:
    matrix = traffic_matrix()
    expected = len(DEFAULT_WORKLOAD_MATRIX) * len(DEFAULT_LOAD_PROFILE_MATRIX) * 3
    assert len(matrix) == expected
    assert matrix_size() == expected


def test_traffic_matrix_custom_repetitions() -> None:
    matrix = traffic_matrix(repetitions=5)
    for cell in matrix:
        assert cell["repetition_total"] == 5
    assert (
        len(matrix)
        == len(DEFAULT_WORKLOAD_MATRIX) * len(DEFAULT_LOAD_PROFILE_MATRIX) * 5
    )


def test_traffic_matrix_custom_workloads() -> None:
    matrix = traffic_matrix(workloads=("random-online", "sharegpt-online"))
    workloads_seen = {cell["workload"] for cell in matrix}
    assert workloads_seen == {"random-online", "sharegpt-online"}


def test_traffic_matrix_rejects_low_repetitions() -> None:
    with pytest.raises(ValueError, match="repetitions must be >= 3"):
        traffic_matrix(repetitions=2)


def test_traffic_matrix_rejects_bad_workload() -> None:
    with pytest.raises(ValueError, match="unsupported workload"):
        traffic_matrix(workloads=("unknown-workload",))


def test_traffic_matrix_rejects_bad_load_profile() -> None:
    with pytest.raises(ValueError, match="unsupported load profile"):
        traffic_matrix(load_profiles=("steady-2rps",))


def test_traffic_matrix_independent_process_per_cell() -> None:
    matrix = traffic_matrix(repetitions=3)
    # Each (workload, profile) cell should have 3 unique repetition indices.
    from collections import defaultdict

    by_cell: dict[tuple, list[int]] = defaultdict(list)
    for cell in matrix:
        by_cell[(cell["workload"], cell["load_profile"])].append(
            cell["repetition_index"]
        )
    for indices in by_cell.values():
        assert indices == [1, 2, 3]


# ---------------------------------------------------------------------------
# write_artifact / load_artifact round-trip tests.
# ---------------------------------------------------------------------------


def test_write_and_load_artifact_round_trip(tmp_path: Path) -> None:
    artifact = _build_artifact()
    path = tmp_path / "artifact.json"
    write_artifact(artifact, path)
    loaded = load_artifact(path)
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["artifact_class"] == ARTIFACT_CLASS


def test_write_artifact_rejects_invalid(tmp_path: Path) -> None:
    artifact = _build_artifact()
    artifact["environment"]["cann_version"] = "unknown"
    with pytest.raises(ValueError, match="cann_version must be explicitly recorded"):
        write_artifact(artifact, tmp_path / "bad.json")


def test_load_artifact_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        load_artifact(path)


# ---------------------------------------------------------------------------
# Edge case: aggregator deep-copies raw_values so caller mutations do not
# corrupt the aggregate.
# ---------------------------------------------------------------------------


def test_aggregate_raw_values_are_independent_of_caller() -> None:
    throughputs = [40.0, 50.0, 60.0]
    artifacts = [
        _build_artifact(rep_index=i, rep_total=3, throughput=t)
        for i, t in enumerate(throughputs, start=1)
    ]
    aggregate = aggregate_repetitions(artifacts)
    raw_values = aggregate["metrics"]["output_throughput_tps"]["raw_values"]
    # Mutating the original list must not affect the aggregate.
    throughputs[0] = 999.0
    assert raw_values == [40.0, 50.0, 60.0]


def test_aggregate_handles_burst_profile_with_burst_recovery() -> None:
    artifacts = [
        _build_artifact(
            rep_index=i,
            rep_total=3,
            profile="burst",
            burst=True,
            throughput=30.0 + i,
        )
        for i in range(1, 4)
    ]
    aggregate = aggregate_repetitions(artifacts)
    assert aggregate["load_profile"] == "burst"
    assert aggregate["overall_status"] == "admitted"
    assert aggregate["metrics"]["output_throughput_tps"]["median"] == 32.0


def test_aggregate_rejects_burst_artifact_with_steady_metrics() -> None:
    """A burst-profile artifact must carry burst_recovery_s; if it does not,
    schema validation rejects it before aggregation."""
    artifacts = []
    for i in range(1, 4):
        artifact = _build_artifact(
            rep_index=i, rep_total=3, profile="burst", burst=True, throughput=30.0 + i
        )
        # Strip burst_recovery_s — should fail semantic validation.
        artifact["slo_metrics"]["burst_recovery_s"] = None
        artifacts.append(artifact)
    with pytest.raises(ValueError, match="burst_recovery_s is required for burst"):
        aggregate_repetitions(artifacts)


def test_aggregate_preserves_workload_and_profile_in_summary() -> None:
    artifacts = _three_repetitions()
    aggregate = aggregate_repetitions(artifacts)
    assert aggregate["workload"] == "random-online"
    assert aggregate["load_profile"] == "steady-1rps"
    assert aggregate["model"] == "Qwen/Qwen2.5-14B-Instruct"
    assert aggregate["hardware_chip_model"] == "910B3"
    assert aggregate["chip_count"] == 1
