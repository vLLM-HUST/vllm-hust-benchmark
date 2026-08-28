"""Tests for the unified metric semantics catalog and resolver.

Covers the ``MetricSemantics`` contract, ``MetricCatalog`` resolution, R1-R5
validation, the catalog-derived metric lists in dependent modules, and the
``--check`` CLI gate (issue #200).
"""

from __future__ import annotations

import pytest

from vllm_hust_benchmark import leaderboard_export
from vllm_hust_benchmark import perfgate_measurement
from vllm_hust_benchmark.metric_semantics import (
    METRIC_CATALOG,
    MetricCatalog,
    MetricDirection,
    MetricRole,
    generate_metric_definitions_strings,
    run_catalog_check,
)

# Canonical names every dependent module is expected to resolve through the
# catalog.  These mirror the previously hardcoded tuples in
# leaderboard_export.py and guard against drift in either direction.
EXPECTED_REQUIRED_METRIC_KEYS = (
    "ttft_ms",
    "throughput_tps",
    "peak_mem_mb",
    "error_rate",
)

EXPECTED_REQUIRED_CONSTRAINT_METRIC_KEYS = (
    "single_chip_effective_utilization_pct",
    "typical_throughput_ratio_vs_baseline",
    "typical_ttft_reduction_pct_vs_baseline",
    "typical_tpot_reduction_pct_vs_baseline",
    "long_context_length",
    "long_context_throughput_stable",
    "long_context_ttft_p95_ms",
    "long_context_ttft_p99_ms",
    "long_context_tpot_p95_ms",
    "long_context_tpot_p99_ms",
    "long_context_ttft_p95_stable",
    "long_context_ttft_p99_stable",
    "long_context_tpot_p95_stable",
    "long_context_tpot_p99_stable",
    "unit_token_cost_reduction_pct",
    "multi_tenant_high_utilization",
)

EXPECTED_DERIVABLE_LONG_CONTEXT_METRIC_KEYS = (
    "long_context_length",
    "long_context_throughput_stable",
    "long_context_ttft_p95_ms",
    "long_context_ttft_p99_ms",
    "long_context_tpot_p95_ms",
    "long_context_tpot_p99_ms",
    "long_context_ttft_p95_stable",
    "long_context_ttft_p99_stable",
    "long_context_tpot_p95_stable",
    "long_context_tpot_p99_stable",
)


class TestCatalogContent:
    def test_required_metric_keys_registered(self) -> None:
        for name in EXPECTED_REQUIRED_METRIC_KEYS:
            assert METRIC_CATALOG.has_name(name), name

    def test_required_constraint_metric_keys_registered(self) -> None:
        for name in EXPECTED_REQUIRED_CONSTRAINT_METRIC_KEYS:
            assert METRIC_CATALOG.has_name(name), name

    def test_client_performance_metrics_order(self) -> None:
        names = [m.name for m in METRIC_CATALOG.client_performance_metrics]
        assert names == ["throughput_tps", "ttft_ms", "tbt_ms"]


class TestResolver:
    def test_resolve_canonical(self) -> None:
        assert METRIC_CATALOG.resolve("ttft_ms").name == "ttft_ms"

    def test_resolve_alias_tpot(self) -> None:
        semantics = METRIC_CATALOG.resolve("tpot_ms")
        assert semantics.name == "tbt_ms"
        assert semantics.direction is MetricDirection.LOWER_IS_BETTER

    def test_resolve_alias_mean_tpot(self) -> None:
        assert METRIC_CATALOG.resolve("mean_tpot_ms").name == "mean_tbt_ms"

    def test_resolve_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            METRIC_CATALOG.resolve("does_not_exist")

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            METRIC_CATALOG.get("does_not_exist")


class TestSemantics:
    def test_throughput_semantics(self) -> None:
        s = METRIC_CATALOG.resolve("throughput_tps")
        assert s.role is MetricRole.PRIMARY
        assert s.direction is MetricDirection.HIGHER_IS_BETTER
        assert s.unit == "tokens/s"

    def test_ttft_mode_applicability(self) -> None:
        s = METRIC_CATALOG.resolve("ttft_ms")
        assert s.is_applicable_for("online") is True
        assert s.is_applicable_for("throughput") is False
        assert s.is_applicable_for("latency") is True

    @pytest.mark.parametrize(
        "name",
        (
            "long_context_throughput_stable",
            "long_context_ttft_p95_stable",
            "long_context_ttft_p99_stable",
            "long_context_tpot_p95_stable",
            "long_context_tpot_p99_stable",
        ),
    )
    def test_boolean_stability_flags_prefer_true(self, name: str) -> None:
        semantics = METRIC_CATALOG.resolve(name)
        assert semantics.unit == "boolean"
        assert semantics.direction is MetricDirection.HIGHER_IS_BETTER
        assert semantics.precision == 0


class TestValidation:
    def test_r1_r5_pass(self) -> None:
        assert METRIC_CATALOG.validate() == []

    def test_run_catalog_check_ok(self, capsys: pytest.CaptureFixture) -> None:
        assert run_catalog_check() == 0
        captured = capsys.readouterr()
        assert "validation OK" in captured.out

    def test_catalog_can_be_frozen(self) -> None:
        catalog = MetricCatalog()
        catalog.freeze()
        semantics = catalog.resolve("ttft_ms")
        with pytest.raises(RuntimeError):
            catalog._register(semantics)

    def test_duplicate_registration_rejected(self) -> None:
        # A fresh catalog registers builtins then freezes itself, so any
        # re-registration of an existing metric is rejected (either by the
        # duplicate-name guard or by the frozen-catalog guard).
        catalog = MetricCatalog()
        with pytest.raises((ValueError, RuntimeError)):
            catalog._register(catalog.resolve("ttft_ms"))


class TestLeaderboardDerivation:
    def test_required_metric_keys(self) -> None:
        assert leaderboard_export.REQUIRED_METRIC_KEYS == EXPECTED_REQUIRED_METRIC_KEYS

    def test_required_constraint_metric_keys(self) -> None:
        assert (
            leaderboard_export.REQUIRED_CONSTRAINT_METRIC_KEYS
            == EXPECTED_REQUIRED_CONSTRAINT_METRIC_KEYS
        )

    def test_derivable_long_context_metric_keys(self) -> None:
        assert (
            leaderboard_export.DERIVABLE_LONG_CONTEXT_METRIC_KEYS
            == EXPECTED_DERIVABLE_LONG_CONTEXT_METRIC_KEYS
        )

    def test_perfgate_measurement_derives_from_catalog(self) -> None:
        client = [m.name for m in METRIC_CATALOG.client_performance_metrics]
        assert perfgate_measurement.PERFORMANCE_METRICS == tuple(client)
        assert perfgate_measurement.SELECTED_RUN_METRICS == (*client, "error_rate")
        assert perfgate_measurement.PER_RUN_METRICS == (
            *client,
            "error_rate",
            "peak_mem_mb",
        )


class TestReportDefinitions:
    def test_definitions_strings_direction_phrase(self) -> None:
        md = generate_metric_definitions_strings(["ttft_ms", "throughput_tps"])
        assert "lower is better" in md["ttft_ms"]
        assert "higher is better" in md["throughput_tps"]

    def test_definitions_alias_key_preserved(self) -> None:
        md = generate_metric_definitions_strings(["tpot_ms"])
        assert set(md) == {"tpot_ms"}
        assert "lower is better" in md["tpot_ms"]
