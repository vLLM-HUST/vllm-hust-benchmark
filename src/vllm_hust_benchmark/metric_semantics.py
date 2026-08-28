"""Unified metric semantics contract, catalog, and resolver.

This module is the single authoritative source of truth for all benchmark
metrics in the vllm-hust-benchmark evaluation pipeline.  Every module that
needs to know how a metric is interpreted (direction, unit, display name,
aggregation, role, alias, mode applicability) MUST read from this catalog
rather than hardcoding its own version.

Design follows EvalScope's ``MetricSemantics`` + ``MetricCatalog`` +
``MetricResolver`` pattern, with R1-R5 validation rules for consistency.
"""

from __future__ import annotations

import argparse
import enum
import sys


# ──────────────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────────────


class MetricRole(enum.Enum):
    """Role of a metric within a benchmark evaluation."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    CONSTRAINT = "constraint"


class MetricDirection(enum.Enum):
    """Which direction is considered better for this metric."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


# ──────────────────────────────────────────────────────────────────────
# Semantics contract
# ──────────────────────────────────────────────────────────────────────


class MetricSemantics:
    """Immutable contract describing one benchmark metric.

    Parameters
    ----------
    name:
        Canonical name (e.g. ``"throughput_tps"``).
    role:
        Role in the evaluation (primary / secondary / constraint).
    direction:
        Which direction is better.
    unit:
        Human-readable unit string (e.g. ``"ms"``, ``"tokens/s"``, ``"MB"``).
    display_name:
        Short human-readable label (e.g. ``"Throughput (tok/s)"``).
    description:
        Longer description of what this metric measures.
    scale:
        Decimal scaling factor (default 1).
    precision:
        Number of decimal places for display (default 2).
    aggregation:
        Aggregation method if derived (e.g. ``"mean"``, ``"median"``,
        ``"p95"``, ``"p99"``).  ``None`` means the raw metric.
    aliases:
        Alternative names that resolve to this canonical metric (e.g.
        ``("tpot_ms",)`` for ``tbt_ms``).
    mode_applicability:
        Optional mapping of workload-mode (e.g. ``"online"``,
        ``"throughput"``, ``"latency"``) to a boolean indicating whether
        this metric is meaningful.
    """

    def __init__(
        self,
        *,
        name: str,
        role: MetricRole,
        direction: MetricDirection,
        unit: str,
        display_name: str,
        description: str,
        scale: int = 1,
        precision: int = 2,
        aggregation: str | None = None,
        aliases: tuple[str, ...] = (),
        mode_applicability: dict[str, bool] | None = None,
    ) -> None:
        self._name = name
        self._role = role
        self._direction = direction
        self._unit = unit
        self._display_name = display_name
        self._description = description
        self._scale = scale
        self._precision = precision
        self._aggregation = aggregation
        self._aliases = aliases
        self._mode_applicability = mode_applicability

    # ── read-only properties ──────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def role(self) -> MetricRole:
        return self._role

    @property
    def direction(self) -> MetricDirection:
        return self._direction

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def display_name(self) -> str:
        return self._display_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def scale(self) -> int:
        return self._scale

    @property
    def precision(self) -> int:
        return self._precision

    @property
    def aggregation(self) -> str | None:
        return self._aggregation

    @property
    def aliases(self) -> tuple[str, ...]:
        return self._aliases

    @property
    def mode_applicability(self) -> dict[str, bool] | None:
        return self._mode_applicability

    # ── helpers ───────────────────────────────────────────────────

    def is_applicable_for(self, mode: str) -> bool:
        """Whether this metric is meaningful in the given workload mode."""
        if self._mode_applicability is None:
            return True
        return self._mode_applicability.get(mode, True)

    def to_report_dict(self) -> dict[str, str]:
        """Serialise to the ``metric_definitions`` dict used in reports."""
        parts = [f"role={self._role.value}"]
        parts.append(f"direction={self._direction.value}")
        parts.append(f"unit={self._unit}")
        if self._aggregation:
            parts.append(f"aggregation={self._aggregation}")
        if self._aliases:
            parts.append(f"aliases={','.join(self._aliases)}")
        return {
            "name": self._name,
            "display_name": self._display_name,
            "description": self._description,
            "semantics": "; ".join(parts),
        }

    def __repr__(self) -> str:
        return (
            f"MetricSemantics(name={self._name!r}, "
            f"role={self._role.value}, "
            f"direction={self._direction.value})"
        )


# ──────────────────────────────────────────────────────────────────────
# Catalog — the single authoritative source
# ──────────────────────────────────────────────────────────────────────


class MetricCatalog:
    """Registry of all benchmark metrics.

    This is the **only** place where metric semantics are defined.  All
    other modules must import and use this catalog.
    """

    def __init__(self) -> None:
        self._by_name: dict[str, MetricSemantics] = {}
        self._alias_map: dict[str, str] = {}
        self._frozen = False
        self._register_builtins()

    def _register(self, m: MetricSemantics) -> None:
        if self._frozen:
            raise RuntimeError("MetricCatalog is frozen")
        # canonical name
        if m.name in self._by_name:
            raise ValueError(f"duplicate metric name: {m.name}")
        self._by_name[m.name] = m
        # aliases
        for alias in m.aliases:
            if alias in self._alias_map:
                raise ValueError(
                    f"alias {alias!r} already points to {self._alias_map[alias]!r}"
                )
            if alias in self._by_name:
                raise ValueError(f"alias {alias!r} conflicts with canonical name")
            self._alias_map[alias] = m.name

    def freeze(self) -> None:
        """Prevent further registration (ensures immutability at runtime)."""
        self._frozen = True

    def get(self, name: str) -> MetricSemantics:
        """Look up by canonical name (raises KeyError if not found)."""
        return self._by_name[name]

    def resolve(self, name: str) -> MetricSemantics:
        """Resolve any name (canonical or alias) to the full semantics.

        Raises ``KeyError`` if the name is unknown.
        """
        if name in self._by_name:
            return self._by_name[name]
        canonical = self._alias_map.get(name)
        if canonical is not None:
            return self._by_name[canonical]
        raise KeyError(f"unknown metric: {name!r}")

    def has_name(self, name: str) -> bool:
        """Check if a name (canonical or alias) is known."""
        return name in self._by_name or name in self._alias_map

    @property
    def all_metrics(self) -> dict[str, MetricSemantics]:
        """All canonical metrics (name → semantics)."""
        return dict(self._by_name)

    @property
    def primary_metrics(self) -> list[MetricSemantics]:
        """Metrics with role=PRIMARY."""
        return [m for m in self._by_name.values() if m.role == MetricRole.PRIMARY]

    @property
    def performance_metrics(self) -> list[MetricSemantics]:
        """Metrics with role=PRIMARY or role=SECONDARY."""
        return [
            m
            for m in self._by_name.values()
            if m.role in (MetricRole.PRIMARY, MetricRole.SECONDARY)
        ]

    @property
    def client_performance_metrics(self) -> list[MetricSemantics]:
        """Canonical client-measured performance metrics.

        These are the three metrics measured per client run and published on
        the leaderboard: output throughput (primary), TTFT (primary) and
        inter-token latency ``tbt_ms``/``tpot_ms`` (secondary).  Perfgate and
        measurement modules derive their metric lists from this set so names
        never drift out of the catalog.
        """
        names = ("throughput_tps", "ttft_ms", "tbt_ms")
        return [self._by_name[name] for name in names]

    @property
    def constraint_metrics(self) -> list[MetricSemantics]:
        """Metrics with role=CONSTRAINT."""
        return [m for m in self._by_name.values() if m.role == MetricRole.CONSTRAINT]

    def validate(self) -> list[str]:
        """Run R1-R5 validation rules and return a list of issues.

        Returns an empty list if all checks pass.
        """
        issues: list[str] = []

        # R1: All metric names must be unique (enforced by _register, but
        #     re-verified here so validation is self-contained).
        seen: set[str] = set()
        for name in self._by_name:
            if name in seen:
                issues.append(f"R1 fail: duplicate canonical metric name: {name!r}")
            seen.add(name)

        # R2: All metric names must be resolvable — every alias must point to
        #     an existing canonical metric (no dangling aliases).
        for alias, canonical in self._alias_map.items():
            if canonical not in self._by_name:
                issues.append(
                    f"R2 fail: alias {alias!r} points to unknown canonical "
                    f"{canonical!r}"
                )

        # R3: No direction/unit drift within a metric family.  Aliases
        #     trivially share semantics, but derived/nearby metrics measuring
        #     the same physical quantity must keep a consistent "which
        #     direction is better" AND "which unit" contract.  Relative-change
        #     metrics (*_reduction_pct_vs_baseline, *_ratio_vs_baseline) and
        #     boolean stability flags are deliberately excluded: they are a
        #     different quantity where "more is better" even for a latency
        #     reduction, so their direction intentionally differs.
        families = {
            "ttft": {
                "ttft_ms",
                "mean_ttft_ms",
                "avg_latency",
                "long_context_ttft_p95_ms",
                "long_context_ttft_p99_ms",
            },
            "tpot": {
                "tbt_ms",
                "mean_tbt_ms",
                "p95_tpot_ms",
                "p99_tpot_ms",
                "long_context_tpot_p95_ms",
                "long_context_tpot_p99_ms",
            },
            "throughput": {
                "throughput_tps",
            },
        }
        family_expectation = {
            "ttft": (MetricDirection.LOWER_IS_BETTER, "ms"),
            "tpot": (MetricDirection.LOWER_IS_BETTER, "ms"),
            "throughput": (MetricDirection.HIGHER_IS_BETTER, "tokens/s"),
        }
        for family, (expected_direction, expected_unit) in family_expectation.items():
            members = families[family] & set(self._by_name)
            for name in members:
                semantics = self._by_name[name]
                if semantics.direction != expected_direction:
                    issues.append(
                        f"R3 fail: {family} family metric {name!r} has "
                        f"direction {semantics.direction.value!r}, expected "
                        f"{expected_direction.value!r}"
                    )
                if semantics.unit != expected_unit:
                    issues.append(
                        f"R3 fail: {family} family metric {name!r} has unit "
                        f"{semantics.unit!r}, expected {expected_unit!r}"
                    )

        # R4: No duplicate declarations — a name must not be both a canonical
        #     metric and an alias, and no alias may be re-declared.
        for alias in self._alias_map:
            if alias in self._by_name:
                issues.append(f"R4 fail: name {alias!r} is both canonical and alias")

        # R5: Mode-applicability must be consistent within a metric family.
        #     If TTFT is not meaningful in a mode, every TTFT-family metric
        #     must agree.
        if "ttft_ms" in self._by_name:
            reference = self._by_name["ttft_ms"]
            if reference.mode_applicability is not None:
                for mode, applicable in reference.mode_applicability.items():
                    for name in families["ttft"] & set(self._by_name):
                        if name == "ttft_ms":
                            continue
                        semantics = self._by_name[name]
                        if semantics.mode_applicability is not None:
                            if (
                                semantics.mode_applicability.get(mode, True)
                                != applicable
                            ):
                                issues.append(
                                    f"R5 fail: {name!r} mode applicability for "
                                    f"{mode!r} differs from ttft_ms"
                                )

        return issues

    # ── built-in metrics ──────────────────────────────────────────

    def _register_builtins(self) -> None:
        self._register(
            MetricSemantics(
                name="throughput_tps",
                role=MetricRole.PRIMARY,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="tokens/s",
                display_name="Throughput (tok/s)",
                description="Output throughput in tokens per second",
                precision=2,
                aliases=(
                    "output_throughput",
                    "tokens_per_second",
                    "total_token_throughput",
                    "requests_per_second",
                    "request_throughput",
                ),
            )
        )
        self._register(
            MetricSemantics(
                name="ttft_ms",
                role=MetricRole.PRIMARY,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="TTFT (ms)",
                description="Time to First Token in milliseconds",
                precision=2,
                mode_applicability={
                    "online": True,
                    "throughput": False,  # offline throughput does not measure TTFT
                    "latency": True,
                },
            )
        )
        self._register(
            MetricSemantics(
                name="tbt_ms",
                role=MetricRole.SECONDARY,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="TBT (ms)",
                description=(
                    "Time Between Tokens (inter-token latency) in milliseconds; "
                    "also known as TPOT (Time Per Output Token)"
                ),
                precision=2,
                aliases=("tpot_ms",),
                mode_applicability={
                    "online": True,
                    "throughput": True,
                    "latency": True,
                },
            )
        )
        self._register(
            MetricSemantics(
                name="error_rate",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ratio",
                display_name="Error Rate",
                description="Fraction of failed requests (0.0 = no errors)",
                precision=4,
            )
        )
        self._register(
            MetricSemantics(
                name="peak_mem_mb",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="MB",
                display_name="Peak Memory (MB)",
                description="Peak memory usage in megabytes",
                precision=0,
            )
        )
        # ── derived / constraint metrics for leaderboard ──────────
        self._register(
            MetricSemantics(
                name="mean_ttft_ms",
                role=MetricRole.SECONDARY,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="Mean TTFT (ms)",
                description="Mean Time to First Token (mean aggregation)",
                aggregation="mean",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="mean_tbt_ms",
                role=MetricRole.SECONDARY,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="Mean TBT (ms)",
                description="Mean Time Between Tokens (mean aggregation)",
                aggregation="mean",
                precision=2,
                aliases=("mean_tpot_ms",),
            )
        )
        self._register(
            MetricSemantics(
                name="p95_tpot_ms",
                role=MetricRole.SECONDARY,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="TPOT P95 (ms)",
                description="95th percentile Time Per Output Token",
                aggregation="p95",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="p99_tpot_ms",
                role=MetricRole.SECONDARY,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="TPOT P99 (ms)",
                description="99th percentile Time Per Output Token",
                aggregation="p99",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="avg_latency",
                role=MetricRole.SECONDARY,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="Avg Latency (ms)",
                description=(
                    "Average end-to-end latency (deprecated; maps to ttft_ms "
                    "for backward compatibility)"
                ),
                precision=2,
                mode_applicability={
                    "online": True,
                    "throughput": False,
                    "latency": True,
                },
            )
        )
        # ── constraint / SLO metrics ──────────────────────────────
        self._register(
            MetricSemantics(
                name="single_chip_effective_utilization_pct",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="%",
                display_name="Single-Chip Utilisation (%)",
                description="Effective single-chip utilisation percentage",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="typical_throughput_ratio_vs_baseline",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="ratio",
                display_name="Throughput Ratio vs Baseline",
                description="Ratio of throughput relative to the baseline target",
                precision=4,
            )
        )
        self._register(
            MetricSemantics(
                name="typical_ttft_reduction_pct_vs_baseline",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="%",
                display_name="TTFT Reduction vs Baseline (%)",
                description="Percentage reduction in TTFT relative to the baseline target",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="typical_tpot_reduction_pct_vs_baseline",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="%",
                display_name="TPOT Reduction vs Baseline (%)",
                description="Percentage reduction in TPOT relative to the baseline target",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_length",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="tokens",
                display_name="Long Context Length",
                description="Maximum stable context length",
                precision=0,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_throughput_stable",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="boolean",
                display_name="Long Context Throughput (stable)",
                description="Whether throughput is stable under long-context load",
                precision=0,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_ttft_p95_ms",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="Long Context TTFT P95 (ms)",
                description="95th percentile TTFT under long-context load",
                aggregation="p95",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_ttft_p99_ms",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="Long Context TTFT P99 (ms)",
                description="99th percentile TTFT under long-context load",
                aggregation="p99",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_tpot_p95_ms",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="Long Context TPOT P95 (ms)",
                description="95th percentile TPOT under long-context load",
                aggregation="p95",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_tpot_p99_ms",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.LOWER_IS_BETTER,
                unit="ms",
                display_name="Long Context TPOT P99 (ms)",
                description="99th percentile TPOT under long-context load",
                aggregation="p99",
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_ttft_p95_stable",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="boolean",
                display_name="Long Context TTFT P95 Stable",
                description="Whether long-context TTFT P95 is stable",
                precision=0,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_ttft_p99_stable",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="boolean",
                display_name="Long Context TTFT P99 Stable",
                description="Whether long-context TTFT P99 is stable",
                precision=0,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_tpot_p95_stable",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="boolean",
                display_name="Long Context TPOT P95 Stable",
                description="Whether long-context TPOT P95 is stable",
                precision=0,
            )
        )
        self._register(
            MetricSemantics(
                name="long_context_tpot_p99_stable",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="boolean",
                display_name="Long Context TPOT P99 Stable",
                description="Whether long-context TPOT P99 is stable",
                precision=0,
            )
        )
        self._register(
            MetricSemantics(
                name="unit_token_cost_reduction_pct",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="%",
                display_name="Unit Token Cost Reduction (%)",
                description=(
                    "Percentage reduction in per-token cost relative to the "
                    "baseline target"
                ),
                precision=2,
            )
        )
        self._register(
            MetricSemantics(
                name="multi_tenant_high_utilization",
                role=MetricRole.CONSTRAINT,
                direction=MetricDirection.HIGHER_IS_BETTER,
                unit="boolean",
                display_name="Multi-Tenant High Utilization",
                description="Whether multi-tenant deployment sustains high utilization",
                precision=0,
            )
        )
        self.freeze()


# ──────────────────────────────────────────────────────────────────────
# Module-level singleton (import this to access the catalog)
# ──────────────────────────────────────────────────────────────────────

METRIC_CATALOG = MetricCatalog()


# ──────────────────────────────────────────────────────────────────────
# Convenience: generate metric_definitions for report emission
# ──────────────────────────────────────────────────────────────────────


def generate_metric_definitions(
    metric_names: list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Generate the ``metric_definitions`` dict used in issue reports.

    When ``metric_names`` is ``None``, all performance metrics are included.
    """
    catalog = METRIC_CATALOG
    if metric_names is None:
        metrics = catalog.performance_metrics
    else:
        metrics = [catalog.resolve(n) for n in metric_names]
    return {m.name: m.to_report_dict() for m in metrics}


def generate_metric_definitions_strings(
    metric_names: list[str] | None = None,
) -> dict[str, str]:
    """Generate the legacy ``metric_definitions`` string map for issue reports.

    The legacy report format keys each metric by the requested name (aliases
    are preserved as-is, e.g. ``tpot_ms``) and maps it to a human-readable
    ``"<description> (higher/lower is better)"`` string built from the
    catalog contract, so the direction phrase never drifts out of sync.

    When ``metric_names`` is ``None``, all performance metrics are included.
    """
    catalog = METRIC_CATALOG
    if metric_names is None:
        names = [m.name for m in catalog.performance_metrics]
    else:
        names = list(metric_names)
    return {n: _metric_definition_string(catalog.resolve(n)) for n in names}


def _metric_definition_string(semantics: MetricSemantics) -> str:
    if semantics.direction is MetricDirection.HIGHER_IS_BETTER:
        phrase = "higher is better"
    else:
        phrase = "lower is better"
    return f"{semantics.description} ({phrase})"


# ──────────────────────────────────────────────────────────────────────
# CLI entry point — R1-R5 validation gate
# ──────────────────────────────────────────────────────────────────────


def run_catalog_check() -> int:
    """Validate the catalog (R1-R5) and return a process exit code.

    ``0`` means the catalog is consistent; ``1`` means one or more rules
    failed.  This is the fail-closed gate CI runs so future metric drift
    (duplicate definitions, dangling aliases, direction/unit drift, mode
    inconsistencies) fails loudly instead of silently propagating.
    """
    issues = METRIC_CATALOG.validate()
    for issue in issues:
        print(f"METRIC CATALOG ERROR: {issue}", file=sys.stderr)
    if issues:
        print(
            f"metric catalog validation FAILED with {len(issues)} issue(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"metric catalog validation OK ({len(METRIC_CATALOG.all_metrics)} metrics, "
        f"{len(METRIC_CATALOG.constraint_metrics)} constraints)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vllm_hust_benchmark.metric_semantics",
        description="Validate the unified metric semantics catalog (R1-R5).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run R1-R5 validation and exit non-zero on any drift",
    )
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("--check is required")
    return run_catalog_check()


if __name__ == "__main__":
    raise SystemExit(main())
