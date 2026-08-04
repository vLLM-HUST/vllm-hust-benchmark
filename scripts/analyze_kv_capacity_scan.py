#!/usr/bin/env python3
"""Analyze KV capacity scan and tiering state machine results for issue #134.

This module processes raw benchmark results from the KV capacity scan
(``scripts/kv_capacity_scan.sh``) and produces structured analysis:

- Per-repetition metric extraction from ``raw.json`` (vLLM bench serve output)
- Aggregation across repetitions (median, IQR, mean, std)
- Capacity curve analysis: identify throughput/latency inflection points
- Tiering comparison: contrast HBM-only vs tiering-disabled vs tiering-enabled
- Preempt timeline analysis from parsed server logs
- Final report generation with issue #134 acceptance criteria check

PR #146 review fixes:
  - Real tiering configs (hbm-only / tiering-disabled / tiering-enabled) instead
    of fake gpu_memory_utilization-only switching.
  - ``check_acceptance_criteria`` no longer fail-open: missing evidence is
    reported as ``blocked`` or ``incomplete`` rather than ``negative-result``.
  - Provenance and repetition validation gate acceptance.
  - Timeline criterion checks ``timeline_complete`` per episode, not just
    ``total_preemptions > 0``.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants from issue #134 acceptance criteria
# ---------------------------------------------------------------------------

KV_CAPACITY_TARGETS_GIB = [8, 16, 24, 32]
SCAN_WORKLOADS = ["random-online", "sharegpt-online", "prefix-repetition-online"]
MIN_REPETITIONS = 3

# Real tiering configs for Part B:
#   hbm-only         — 32 GiB KV, no kv-transfer-config (baseline, no pressure)
#   tiering-disabled — 8 GiB KV, no kv-transfer-config (pressure, no tiering)
#   tiering-enabled  — 8 GiB KV, CPUOffloadingConnector (pressure + tiering)
REQUIRED_TIERING_CONFIGS = ["hbm-only", "tiering-disabled", "tiering-enabled"]
# Backward-compatible alias
TIERING_CONFIGS = REQUIRED_TIERING_CONFIGS

# Provenance fields that must be present for evidence to be admissible.
REQUIRED_PROVENANCE_FIELDS = [
    "engine_commit",
    "plugin_commit",
    "cann_version",
    "driver_version",
    "torch_npu_version",
    "model_revision",
    "resolved_parameters",
    "actual_kv_bytes",
]

# Metric field names in vLLM bench serve raw.json output
METRIC_FIELDS = {
    "mean_ttft_ms": "mean_ttft_ms",
    "median_ttft_ms": "median_ttft_ms",
    "p99_ttft_ms": "p99_ttft_ms",
    "mean_tpot_ms": "mean_tpot_ms",
    "p99_tpot_ms": "p99_tpot_ms",
    "mean_itl_ms": "mean_itl_ms",
    "p99_itl_ms": "p99_itl_ms",
    "output_throughput": "output_throughput",
    "request_throughput": "request_throughput",
    "max_concurrent_requests": "max_concurrent_requests",
}


def compute_stats(values: list[float]) -> dict[str, float | None]:
    """Compute median, IQR, mean, std, min, max for a list of values.

    Returns a dict with keys: ``median``, ``p25``, ``p75``, ``iqr``,
    ``mean``, ``stdev``, ``min``, ``max``, ``count``.
    """
    if not values:
        return {
            "median": None,
            "p25": None,
            "p75": None,
            "iqr": None,
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
            "count": 0,
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median = statistics.median(sorted_vals)
    p25 = _percentile(sorted_vals, 25)
    p75 = _percentile(sorted_vals, 75)

    return {
        "median": median,
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25 if p25 is not None and p75 is not None else None,
        "mean": statistics.mean(sorted_vals) if n > 0 else None,
        "stdev": statistics.stdev(sorted_vals) if n > 1 else 0.0,
        "min": min(sorted_vals),
        "max": max(sorted_vals),
        "count": n,
    }


def _percentile(sorted_vals: list[float], pct: float) -> float | None:
    """Compute percentile from a pre-sorted list (linear interpolation)."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def extract_metrics(raw_result: dict[str, Any]) -> dict[str, float | None]:
    """Extract key metrics from a vLLM bench serve ``raw.json`` result.

    Returns a dict mapping metric names to values. Missing metrics are None.
    """
    return {name: raw_result.get(field) for name, field in METRIC_FIELDS.items()}


def aggregate_reps(
    reps: list[dict[str, Any]],
) -> dict[str, dict[str, float | None] | dict[str, float | None]]:
    """Aggregate metric values across multiple repetition results.

    Args:
        reps: list of raw benchmark result dicts (each from raw.json)

    Returns:
        Dict with two keys:
        - ``per_metric_stats``: dict mapping metric name to compute_stats output
        - ``raw_values``: dict mapping metric name to list of float values
    """
    all_metrics: dict[str, list[float]] = {name: [] for name in METRIC_FIELDS}

    for rep in reps:
        metrics = extract_metrics(rep)
        for name, val in metrics.items():
            if val is not None:
                all_metrics[name].append(float(val))

    return {
        "per_metric_stats": {
            name: compute_stats(vals) for name, vals in all_metrics.items()
        },
        "raw_values": all_metrics,
    }


def analyze_capacity_scan(
    results: dict[str, dict[str, dict[str, list[dict[str, Any]]]]],
) -> dict[str, Any]:
    """Analyze the full KV capacity scan results.

    Args:
        results: Nested dict ``{workload: {kv_gib: {rep_key: raw_result}}}``

    Returns:
        Analysis dict with:
        - ``capacity_curves``: per-workload, per-capacity aggregated stats
        - ``inflection_points``: detected throughput/latency inflection points
        - ``workloads_covered``: list of workloads analyzed
        - ``capacities_covered``: list of KV capacities analyzed
    """
    capacity_curves: dict[str, Any] = {}
    workloads_covered: list[str] = []
    capacities_covered: set[int] = set()

    for workload, kv_data in results.items():
        workloads_covered.append(workload)
        curve: dict[str, Any] = {}

        for kv_gib_str, reps_dict in kv_data.items():
            kv_gib = int(kv_gib_str)
            capacities_covered.add(kv_gib)
            rep_list = list(reps_dict.values())
            agg = aggregate_reps(rep_list)
            curve[kv_gib_str] = {
                "kv_target_gib": kv_gib,
                "repetitions": len(rep_list),
                "stats": agg["per_metric_stats"],
            }

        capacity_curves[workload] = curve

    inflection = identify_inflection_points(capacity_curves)

    return {
        "capacity_curves": capacity_curves,
        "inflection_points": inflection,
        "workloads_covered": workloads_covered,
        "capacities_covered": sorted(capacities_covered),
    }


def identify_inflection_points(
    capacity_curves: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Identify throughput/latency inflection points across KV capacities.

    An inflection point is the KV capacity where the metric changes most
    significantly (largest absolute delta between consecutive capacities).

    Returns:
        Dict per workload with:
        - ``throughput_inflection_gib``: KV capacity with largest throughput drop
        - ``ttft_inflection_gib``: KV capacity with largest TTFT increase
        - ``p99_ttft_inflection_gib``: KV capacity with largest P99 TTFT increase
    """
    result: dict[str, dict[str, Any]] = {}

    for workload, curve in capacity_curves.items():
        caps = sorted(int(k) for k in curve.keys())
        if len(caps) < 2:
            result[workload] = {
                "throughput_inflection_gib": None,
                "ttft_inflection_gib": None,
                "p99_ttft_inflection_gib": None,
                "note": "insufficient data points",
            }
            continue

        # Extract median values for each capacity
        throughput_vals: list[tuple[int, float | None]] = []
        ttft_vals: list[tuple[int, float | None]] = []
        p99_ttft_vals: list[tuple[int, float | None]] = []

        for cap in caps:
            stats = curve[str(cap)]["stats"]
            throughput_vals.append(
                (cap, stats.get("output_throughput", {}).get("median"))
            )
            ttft_vals.append((cap, stats.get("mean_ttft_ms", {}).get("median")))
            p99_ttft_vals.append((cap, stats.get("p99_ttft_ms", {}).get("median")))

        result[workload] = {
            "throughput_inflection_gib": _find_max_delta_cap(throughput_vals),
            "ttft_inflection_gib": _find_max_delta_cap(ttft_vals),
            "p99_ttft_inflection_gib": _find_max_delta_cap(p99_ttft_vals),
            "throughput_values": [
                {"kv_gib": c, "median": v} for c, v in throughput_vals
            ],
            "ttft_values": [{"kv_gib": c, "median": v} for c, v in ttft_vals],
            "p99_ttft_values": [{"kv_gib": c, "median": v} for c, v in p99_ttft_vals],
        }

    return result


def _find_max_delta_cap(
    cap_vals: list[tuple[int, float | None]],
) -> int | None:
    """Find the capacity where the largest absolute change occurs.

    For throughput (higher is better), the inflection is where the largest
    drop occurs. For latency (lower is better), it's where the largest
    increase occurs. We return the capacity at the end of that transition.
    """
    if len(cap_vals) < 2:
        return None

    max_delta = 0.0
    max_delta_cap: int | None = None

    for i in range(1, len(cap_vals)):
        prev_cap, prev_val = cap_vals[i - 1]
        curr_cap, curr_val = cap_vals[i]
        if prev_val is None or curr_val is None:
            continue
        delta = abs(curr_val - prev_val)
        if delta > max_delta:
            max_delta = delta
            max_delta_cap = curr_cap

    return max_delta_cap


def analyze_tiering_comparison(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Analyze tiering comparison results (Part B of issue #134).

    Args:
        results: Dict ``{config_name: [raw_result, ...]}`` where config_name
            is one of ``hbm-only``, ``tiering-disabled``, ``tiering-enabled``.

    Returns:
        Analysis dict with:
        - ``per_config_stats``: aggregated stats per config
        - ``comparison``: pairwise deltas between configs
        - ``best_config``: config with best throughput/latency
        - ``configs_present``: list of config names with data
        - ``configs_required``: the required config names
        - ``configs_complete``: configs with >= MIN_REPETITIONS reps and
          finite-positive throughput
    """
    per_config: dict[str, Any] = {}
    configs_complete: list[str] = []

    for config, reps in results.items():
        agg = aggregate_reps(reps)
        per_config[config] = {
            "repetitions": len(reps),
            "stats": agg["per_metric_stats"],
        }

        # A config is "complete" if it has >= MIN_REPETITIONS reps with
        # finite-positive output throughput.
        tput_vals = [r.get("output_throughput") for r in reps]
        is_complete = len(reps) >= MIN_REPETITIONS and all(
            v is not None and math.isfinite(float(v)) and float(v) > 0
            for v in tput_vals
        )
        if is_complete:
            configs_complete.append(config)

    comparison: dict[str, Any] = {}
    configs = list(results.keys())

    for i, cfg_a in enumerate(configs):
        for cfg_b in configs[i + 1 :]:
            stats_a = per_config[cfg_a]["stats"]
            stats_b = per_config[cfg_b]["stats"]
            key = f"{cfg_a}_vs_{cfg_b}"

            throughput_a = stats_a.get("output_throughput", {}).get("median")
            throughput_b = stats_b.get("output_throughput", {}).get("median")
            ttft_a = stats_a.get("mean_ttft_ms", {}).get("median")
            ttft_b = stats_b.get("mean_ttft_ms", {}).get("median")

            comparison[key] = {
                "throughput_delta_pct": _delta_pct(throughput_a, throughput_b),
                "ttft_delta_pct": _delta_pct(ttft_a, ttft_b),
            }

    # Determine best config (highest throughput, lowest TTFT)
    best_throughput = None
    best_throughput_cfg = None
    best_ttft = None
    best_ttft_cfg = None

    for cfg, data in per_config.items():
        tput = data["stats"].get("output_throughput", {}).get("median")
        ttft = data["stats"].get("mean_ttft_ms", {}).get("median")
        if tput is not None and (best_throughput is None or tput > best_throughput):
            best_throughput = tput
            best_throughput_cfg = cfg
        if ttft is not None and (best_ttft is None or ttft < best_ttft):
            best_ttft = ttft
            best_ttft_cfg = cfg

    return {
        "per_config_stats": per_config,
        "comparison": comparison,
        "best_config": {
            "throughput": best_throughput_cfg,
            "ttft": best_ttft_cfg,
        },
        "configs_present": list(results.keys()),
        "configs_required": list(REQUIRED_TIERING_CONFIGS),
        "configs_complete": configs_complete,
    }


def _delta_pct(base: float | None, head: float | None) -> float | None:
    """Compute percentage delta: (head - base) / base * 100."""
    if base is None or head is None or base == 0:
        return None
    return round(((head - base) / base) * 100, 2)


def _validate_provenance(
    provenance: dict[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Validate that all required provenance fields are present and non-None.

    Returns:
        Tuple of (is_valid, missing_fields).
    """
    if not isinstance(provenance, dict) or not provenance:
        return False, list(REQUIRED_PROVENANCE_FIELDS)
    missing = [f for f in REQUIRED_PROVENANCE_FIELDS if provenance.get(f) is None]
    return len(missing) == 0, missing


def _validate_repetitions(
    analysis: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Check all formal points have >= MIN_REPETITIONS reps with finite-positive throughput.

    Inspects both capacity scan curves and tiering comparison per-config stats.
    Per reviewer round 1 issue 4: an empty ``capacity_curves`` dict is NOT a
    vacuous pass — it is a blocking failure because there is no evidence to
    validate.

    Returns:
        Tuple of (is_valid, issues) where issues is a list of human-readable
        problem descriptions.
    """
    issues: list[str] = []

    capacity_curves = analysis.get("capacity_curves", {})
    if not capacity_curves:
        issues.append("capacity_curves is empty — no evidence to validate")
    for workload, curve in capacity_curves.items():
        for cap_str, data in curve.items():
            reps = data.get("repetitions", 0)
            if reps < MIN_REPETITIONS:
                issues.append(
                    f"capacity {workload}/{cap_str}: {reps} reps < {MIN_REPETITIONS}"
                )
            # Per reviewer round 1 issue 4: check each rep's throughput, not
            # just the median, to catch NaN/0 hidden behind a positive median.
            raw_values = data.get("raw_values", [])
            for i, v in enumerate(raw_values):
                if v is None or not math.isfinite(v) or v <= 0:
                    issues.append(
                        f"capacity {workload}/{cap_str}: rep {i + 1} invalid "
                        f"throughput {v}"
                    )
            stats = data.get("stats", {})
            median = stats.get("output_throughput", {}).get("median")
            if median is None:
                issues.append(f"capacity {workload}/{cap_str}: no throughput data")
            elif not math.isfinite(median) or median <= 0:
                issues.append(
                    f"capacity {workload}/{cap_str}: invalid throughput {median}"
                )

    tiering = analysis.get("tiering_comparison", {})
    for config, data in tiering.get("per_config_stats", {}).items():
        reps = data.get("repetitions", 0)
        if reps < MIN_REPETITIONS:
            issues.append(f"tiering {config}: {reps} reps < {MIN_REPETITIONS}")
        stats = data.get("stats", {})
        median = stats.get("output_throughput", {}).get("median")
        if median is None:
            issues.append(f"tiering {config}: no throughput data")
        elif not math.isfinite(median) or median <= 0:
            issues.append(f"tiering {config}: invalid throughput {median}")

    return len(issues) == 0, issues


def check_acceptance_criteria(analysis: dict[str, Any]) -> dict[str, Any]:
    """Check issue #134 acceptance criteria against analysis results.

    The overall status is determined as follows:
      - ``admitted``: all criteria met.
      - ``blocked``: missing evidence (capacities, provenance, tiering configs,
        or timeline data absent). Cannot determine a result.
      - ``incomplete``: evidence present but insufficient (too few reps,
        incomplete preempt timeline). Needs more data.
      - ``negative-result``: all evidence present but a criterion is not met
        (e.g., no inflection points found despite complete data).

    Returns a dict with:
    - ``all_criteria_met``: bool
    - ``criteria``: list of per-criterion dicts
    - ``overall_status``: one of the four statuses above
    - ``provenance_validation``: dict from _validate_provenance
    - ``repetition_validation``: dict from _validate_repetitions
    """
    criteria: list[dict[str, Any]] = []
    has_blocking_failure = False
    has_incomplete_failure = False

    # 1. 8/16/24/32 GiB capacity curves produced
    caps = analysis.get("capacities_covered", [])
    caps_met = all(c in caps for c in KV_CAPACITY_TARGETS_GIB)
    if not caps_met:
        has_blocking_failure = True
    criteria.append(
        {
            "criterion": "8/16/24/32 GiB capacity curves produced",
            "met": caps_met,
            "details": f"Covered: {sorted(caps)}, expected: {KV_CAPACITY_TARGETS_GIB}",
        }
    )

    # 2. Clear inflection points identified
    inflection = analysis.get("inflection_points", {})
    has_inflection = any(
        v.get("throughput_inflection_gib") is not None
        or v.get("ttft_inflection_gib") is not None
        for v in inflection.values()
        if isinstance(v, dict)
    )
    # Inflection failure alone → negative-result (evidence is complete,
    # just no inflection found).
    criteria.append(
        {
            "criterion": "Clear service rate/tail latency inflection points identified",
            "met": has_inflection,
            "details": f"Inflection points: {inflection}",
        }
    )

    # 3. At least one complete preempt timeline (6-stage chain)
    preempt_timeline = analysis.get("preempt_timeline", {})
    total_preemptions = preempt_timeline.get("total_preemptions", 0)
    episodes = preempt_timeline.get("pressure_episodes", [])
    has_complete_timeline = any(
        e.get("timeline_complete") for e in episodes if isinstance(e, dict)
    )
    if has_complete_timeline:
        timeline_met = True
        timeline_details = (
            f"{total_preemptions} preemptions, at least one complete 6-stage timeline"
        )
    elif total_preemptions > 0:
        # Preemptions occurred but no episode has a complete 6-stage chain.
        timeline_met = False
        has_incomplete_failure = True
        timeline_details = (
            f"{total_preemptions} preemptions but no complete 6-stage timeline "
            f"(stages missing)"
        )
    else:
        # No preemptions or no timeline evidence at all.
        timeline_met = False
        has_blocking_failure = True
        timeline_details = "No preempt timeline evidence"
    criteria.append(
        {
            "criterion": (
                "At least one complete preempt->restore->requeue/admission "
                "6-stage timeline"
            ),
            "met": timeline_met,
            "details": timeline_details,
        }
    )

    # 4. Tiering comparison completed (all 3 real configs)
    tiering = analysis.get("tiering_comparison", {})
    configs_present = tiering.get(
        "configs_present", list(tiering.get("per_config_stats", {}).keys())
    )
    configs_required = tiering.get("configs_required", REQUIRED_TIERING_CONFIGS)
    configs_complete = tiering.get("configs_complete", [])
    tiering_met = all(c in configs_complete for c in configs_required)
    if not tiering_met:
        if not all(c in configs_present for c in configs_required):
            has_blocking_failure = True
            tiering_details = (
                f"Missing configs: present={configs_present}, "
                f"required={configs_required}"
            )
        else:
            has_incomplete_failure = True
            tiering_details = (
                f"All configs present but not all complete: "
                f"complete={configs_complete}, required={configs_required}"
            )
    else:
        tiering_details = f"All required configs complete: {configs_complete}"
    criteria.append(
        {
            "criterion": (
                "Tiering disabled/enabled/HBM-only comparison completed "
                "(real configs, not gpu_memory_utilization switching)"
            ),
            "met": tiering_met,
            "details": tiering_details,
        }
    )

    # 5. All formal points have >= MIN_REPETITIONS reps with finite-positive throughput
    reps_valid, reps_issues = _validate_repetitions(analysis)
    if not reps_valid:
        has_incomplete_failure = True
    criteria.append(
        {
            "criterion": (
                f"All formal points have >= {MIN_REPETITIONS} independent "
                f"restarts with finite-positive throughput"
            ),
            "met": reps_valid,
            "details": "; ".join(reps_issues) if reps_issues else "All OK",
        }
    )

    # 6. Provenance present and complete
    provenance = analysis.get("provenance", {})
    prov_valid, prov_missing = _validate_provenance(provenance)
    if not prov_valid:
        has_blocking_failure = True
    criteria.append(
        {
            "criterion": "Environment manifest (provenance) complete",
            "met": prov_valid,
            "details": (
                f"Missing fields: {prov_missing}" if prov_missing else "All OK"
            ),
        }
    )

    all_met = all(c["met"] for c in criteria)

    if all_met:
        overall_status = "admitted"
    elif has_blocking_failure:
        overall_status = "blocked"
    elif has_incomplete_failure:
        overall_status = "incomplete"
    else:
        overall_status = "negative-result"

    return {
        "all_criteria_met": all_met,
        "criteria": criteria,
        "overall_status": overall_status,
        "provenance_validation": {
            "valid": prov_valid,
            "missing": prov_missing,
        },
        "repetition_validation": {
            "valid": reps_valid,
            "issues": reps_issues,
        },
    }


def generate_report(
    capacity_scan_analysis: dict[str, Any],
    tiering_analysis: dict[str, Any] | None = None,
    preempt_timeline: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate the final issue #134 analysis report.

    Args:
        capacity_scan_analysis: Output from analyze_capacity_scan
        tiering_analysis: Output from analyze_tiering_comparison (optional)
        preempt_timeline: Output from reconstruct_preempt_timeline (optional)
        provenance: Environment manifest dict (optional) with
            REQUIRED_PROVENANCE_FIELDS filled

    Returns:
        Complete report dict with all analysis sections and acceptance check.
    """
    combined: dict[str, Any] = {**capacity_scan_analysis}
    if tiering_analysis:
        combined["tiering_comparison"] = tiering_analysis
    if preempt_timeline:
        combined["preempt_timeline"] = preempt_timeline
    if provenance:
        combined["provenance"] = provenance

    acceptance = check_acceptance_criteria(combined)

    return {
        "issue": 134,
        "title": "KV capacity scan and preempt-restore-admission state machine analysis",
        "analysis": combined,
        "acceptance_criteria": acceptance,
        "issue_89_linkage": {
            "status": acceptance["overall_status"],
            "note": (
                "Results linked to #89 KV tiering/offload mechanism evidence. "
                "This issue provides explicit capacity scan and state machine "
                "analysis that complements #89's mechanism coverage."
            ),
        },
    }


def main() -> None:
    """CLI entry point: load results and generate analysis report."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze KV capacity scan results for issue #134."
    )
    parser.add_argument(
        "--results-dir",
        default="reports/issue_134_kv_capacity_scan",
        help="Directory containing raw results",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSON report file (default: stdout)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        return

    # Load capacity scan results
    capacity_file = results_dir / "capacity_scan_results.json"
    if capacity_file.exists():
        capacity_data = json.loads(capacity_file.read_text())
    else:
        # Try loading from raw_results directory structure
        capacity_data = _load_from_raw_results(results_dir / "raw_results")

    capacity_analysis = analyze_capacity_scan(capacity_data)

    # Load tiering comparison if available
    tiering_file = results_dir / "tiering_comparison_results.json"
    tiering_analysis = None
    if tiering_file.exists():
        tiering_data = json.loads(tiering_file.read_text())
        tiering_analysis = analyze_tiering_comparison(tiering_data)

    # Load preempt timeline if available
    timeline_file = results_dir / "preempt_timeline.json"
    preempt_timeline = None
    if timeline_file.exists():
        preempt_timeline = json.loads(timeline_file.read_text())

    # Load provenance if available.  Per reviewer round 1 issue 4: the scan
    # script writes env-manifest.json per-rep (under each rep dir), not at the
    # top level.  Try the top-level manifest first, then fall back to the first
    # rep-level manifest found under raw_results/ or tiering/<config>/rep-*/.
    provenance = None
    manifest_file = results_dir / "env-manifest.json"
    if manifest_file.exists():
        provenance = json.loads(manifest_file.read_text())
    else:
        for sub in [
            results_dir / "raw_results",
            results_dir / "tiering",
        ]:
            if not sub.is_dir():
                continue
            for candidate in sub.rglob("env-manifest.json"):
                provenance = json.loads(candidate.read_text())
                break
            if provenance is not None:
                break

    report = generate_report(
        capacity_analysis, tiering_analysis, preempt_timeline, provenance
    )

    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    else:
        print(output)


def _load_from_raw_results(raw_dir: Path) -> dict[str, Any]:
    """Load results from the raw_results directory structure.

    Expected structure:
        raw_results/<workload>/<kv_gib>/rep-<N>/raw.json

    Returns nested dict: ``{workload: {kv_gib: {rep_key: raw_result}}}``
    """
    results: dict[str, Any] = {}
    if not raw_dir.exists():
        return results

    for workload_dir in sorted(raw_dir.iterdir()):
        if not workload_dir.is_dir():
            continue
        workload = workload_dir.name
        results[workload] = {}

        for kv_dir in sorted(workload_dir.iterdir()):
            if not kv_dir.is_dir():
                continue
            kv_gib = kv_dir.name
            results[workload][kv_gib] = {}

            for rep_dir in sorted(kv_dir.iterdir()):
                if not rep_dir.is_dir():
                    continue
                raw_file = rep_dir / "raw.json"
                if raw_file.exists():
                    results[workload][kv_gib][rep_dir.name] = json.loads(
                        raw_file.read_text()
                    )

    return results


if __name__ == "__main__":
    main()
