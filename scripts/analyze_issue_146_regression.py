"""Analyze issue #146 regression re-test results.

Reads raw benchmark results from the re-test directory, computes medians,
and determines whether the two suspected regressions are reproducible.

Issue #146 acceptance criteria:
- sonnet-throughput: 2206f1f7b7 -> 7a63f81e86 should converge within 10%
- random-latency: 2206f1f7b7 -> 83cf83ff20 should not stably exceed 20% higher
- If noise/environment drift: clean corresponding leaderboard points

Usage:
    python analyze_issue_146_regression.py --result-dir /data/issue146-retest-results
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Commits under investigation (issue #146)
ENGINE_COMMITS = ["2206f1f7b7", "7a63f81e86", "83cf83ff20"]

# Workloads with suspected regressions
WORKLOADS = ["sonnet-throughput", "random-latency"]

# Acceptance thresholds from issue #146
SONNET_THROUGHPUT_THRESHOLD_PCT = 10.0
RANDOM_LATENCY_THRESHOLD_PCT = 20.0


def _load_raw_json(path: Path) -> dict[str, Any] | None:
    """Load a raw.json benchmark output file."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _extract_metric(raw: dict[str, Any], workload: str) -> float | None:
    """Extract the primary metric from a raw benchmark output.

    - sonnet-throughput: ``tokens_per_second`` (tokens/s, higher is better)
    - random-latency: ``avg_latency`` converted to ms (lower is better)

    The field names mirror the actual vllm bench output format (verified
    against vllm/benchmarks/latency.py and the throughput subcommand).
    ``leaderboard_export._derive_metrics_from_benchmark_result`` uses the
    same fallback chain when converting raw output to leaderboard metrics.
    """
    if workload == "sonnet-throughput":
        # vllm bench throughput output:
        #   {"tokens_per_second": ..., "requests_per_second": ...}
        # The leaderboard stores this as throughput_tps.
        val = (
            raw.get("tokens_per_second")
            or raw.get("throughput_tps")
            or raw.get("tokens/s")
        )
        if val is None and isinstance(raw.get("throughput"), dict):
            val = raw["throughput"].get("tokens/s")
        return float(val) if val is not None else None
    elif workload == "random-latency":
        # vllm bench latency output:
        #   {"avg_latency": <seconds>, "latencies": [...], "percentiles": {...}}
        # The leaderboard stores this as ttft_ms (avg_latency * 1000).
        # Return in ms for consistency with the leaderboard convention.
        val = raw.get("mean_ttft_ms")
        if val is not None:
            return float(val)
        val = raw.get("ttft_ms")
        if val is not None:
            return float(val)
        avg_latency_s = raw.get("avg_latency")
        if avg_latency_s is not None:
            return float(avg_latency_s) * 1000.0
        # Fallback for other output formats
        val = raw.get("p50")
        if val is not None:
            return float(val)
        return None
    return None


def collect_results(result_dir: Path) -> dict[str, dict[str, list[float]]]:
    """Collect all benchmark results from the re-test directory.

    Returns: {workload: {commit: [metric1, metric2, ...]}}
    """
    results: dict[str, dict[str, list[float]]] = {}
    for workload in WORKLOADS:
        results[workload] = {}
        for commit in ENGINE_COMMITS:
            commit_dir = result_dir / commit / workload
            if not commit_dir.is_dir():
                results[workload][commit] = []
                continue
            metrics: list[float] = []
            for rep_dir in sorted(commit_dir.iterdir()):
                if not rep_dir.is_dir() or not rep_dir.name.startswith("rep-"):
                    continue
                raw = _load_raw_json(rep_dir / "raw.json")
                if raw is None:
                    continue
                metric = _extract_metric(raw, workload)
                if metric is not None:
                    metrics.append(metric)
            results[workload][commit] = metrics
    return results


def compute_medians(
    results: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute median, min, max, and count for each workload/commit.

    Returns: {workload: {commit: {median, min, max, count, values}}}
    """
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    for workload, commit_results in results.items():
        summary[workload] = {}
        for commit, values in commit_results.items():
            if not values:
                summary[workload][commit] = {
                    "median": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                    "values": [],
                }
            else:
                summary[workload][commit] = {
                    "median": round(statistics.median(values), 2),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                    "count": len(values),
                    "values": [round(v, 2) for v in values],
                }
    return summary


def _delta_pct(base: float | None, head: float | None) -> float | None:
    """Compute percentage change from base to head."""
    if base is None or head is None or base == 0:
        return None
    return round(((head - base) / base) * 100, 2)


def analyze_regression(summary: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Analyze whether the suspected regressions are reproducible.

    Issue #146:
    - sonnet-throughput: 2206f1f7b7 -> 7a63f81e86 (throughput drop)
    - random-latency: 2206f1f7b7 -> 83cf83ff20 (latency increase)

    Acceptance:
    - sonnet: delta should converge within 10% (not a real regression)
    - random-latency: delta should not stably exceed 20% (not a real regression)
    """
    findings: dict[str, Any] = {}

    # sonnet-throughput regression check
    sonnet_base = (
        summary.get("sonnet-throughput", {}).get("2206f1f7b7", {}).get("median")
    )
    sonnet_head = (
        summary.get("sonnet-throughput", {}).get("7a63f81e86", {}).get("median")
    )
    sonnet_delta = _delta_pct(sonnet_base, sonnet_head)

    sonnet_reproducible = False
    if sonnet_delta is not None and sonnet_delta < -SONNET_THROUGHPUT_THRESHOLD_PCT:
        sonnet_reproducible = True

    findings["sonnet-throughput"] = {
        "base_commit": "2206f1f7b7",
        "head_commit": "7a63f81e86",
        "base_median": sonnet_base,
        "head_median": sonnet_head,
        "delta_pct": sonnet_delta,
        "threshold_pct": SONNET_THROUGHPUT_THRESHOLD_PCT,
        "regression_reproducible": sonnet_reproducible,
        "conclusion": (
            "regression_confirmed" if sonnet_reproducible else "no_regression_or_noise"
        ),
    }

    # random-latency regression check
    latency_base = summary.get("random-latency", {}).get("2206f1f7b7", {}).get("median")
    latency_head = summary.get("random-latency", {}).get("83cf83ff20", {}).get("median")
    latency_delta = _delta_pct(latency_base, latency_head)

    latency_reproducible = False
    if latency_delta is not None and latency_delta > RANDOM_LATENCY_THRESHOLD_PCT:
        latency_reproducible = True

    findings["random-latency"] = {
        "base_commit": "2206f1f7b7",
        "head_commit": "83cf83ff20",
        "base_median": latency_base,
        "head_median": latency_head,
        "delta_pct": latency_delta,
        "threshold_pct": RANDOM_LATENCY_THRESHOLD_PCT,
        "regression_reproducible": latency_reproducible,
        "conclusion": (
            "regression_confirmed" if latency_reproducible else "no_regression_or_noise"
        ),
    }

    # Overall conclusion
    any_confirmed = sonnet_reproducible or latency_reproducible
    findings["overall"] = {
        "any_regression_confirmed": any_confirmed,
        "action": ("bisect_and_fix" if any_confirmed else "clean_leaderboard_points"),
        "issue_146_resolution": (
            "Regression(s) reproduced. Bisect to find root cause."
            if any_confirmed
            else "No reproducible regression. Clean suspect leaderboard points "
            "and replace with re-test medians."
        ),
    }

    return findings


def generate_report(
    summary: dict[str, dict[str, dict[str, Any]]],
    findings: dict[str, Any],
    result_dir: Path,
) -> dict[str, Any]:
    """Generate the full analysis report."""
    return {
        "schema_version": "issue-146-retest-report/v1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_dir": str(result_dir),
        "engine_commits": ENGINE_COMMITS,
        "workloads": WORKLOADS,
        "results_summary": summary,
        "regression_analysis": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("/data/issue146-retest-results"),
        help="Directory containing re-test results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file (default: stdout)",
    )
    args = parser.parse_args()

    result_dir: Path = args.result_dir
    if not result_dir.is_dir():
        print(f"ERROR: {result_dir} not found", file=sys.stderr)
        return 2

    results = collect_results(result_dir)
    summary = compute_medians(results)
    findings = analyze_regression(summary)
    report = generate_report(summary, findings, result_dir)

    output_json = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output_json + "\n", encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
