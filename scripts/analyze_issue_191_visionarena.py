#!/usr/bin/env python3
"""Analyze issue #191 regression re-test results (visionarena-online).

Reads raw.json benchmark results from the retest_issue_191_visionarena.sh
output directory, computes median metrics across interleaved repetitions,
and compares base vs head commits for the visionarena-online workload to
determine whether the reported TPOT +71.4% jump is a reproducible regression
or original-data/environment noise.

Simplified single-interval version of analyze_issue_151_regression.py.

Usage:
    python scripts/analyze_issue_191_visionarena.py \\
        --result-dir /data/issue191-retest-results \\
        [--engine-repo /root/vllm/vllm-hust] \\
        [--plugin-repo /root/vllm/vllm-ascend-hust] \\
        [--output report.json]

Acceptance thresholds (from issue #151/#165, same contract):
    - TTFT increase  > 20% = reproducible regression
    - TPOT increase  > 20% = reproducible regression
    - Throughput decrease > 10% = reproducible regression
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.metric_semantics import generate_metric_definitions_strings

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RESULT_DIR = "/data/issue191-retest-results"

# Single interval definition for issue #191. Provenance contract carried over
# from the issue #151/#177 tracking entry (tracking_issue #191).
INTERVAL = {
    "name": "visionarena-online",
    "base_commit": "ec4847981f",
    "head_commit": "ceec19abb0",
    "workload": "visionarena-online",
    "reported_jump": "71.4% TPOT jump",
    "original_leaderboard": {
        "base": {"ttft_ms": 369, "tpot_ms": 35.2, "throughput_tps": 117.0},
        "head": {"ttft_ms": 387, "tpot_ms": 60.3, "throughput_tps": 118.2},
    },
    "metric_definitions": generate_metric_definitions_strings(
        ["ttft_ms", "tpot_ms", "throughput_tps"]
    ),
    "hardware": {"chip_model": "910B2", "chip_count": 1, "node_count": 1},
    "model": {"name": "Qwen2.5-VL-7B-Instruct", "precision": "float16"},
    "same_spec_identity": (
        "resolved_spec_hash matched between base and head (per issue #151); "
        "spec_id=official-ascend-jan-2026-v0.18.0-"
        "visionarena-online-qwen25-vl-7b-910b2"
    ),
    "server_config": {
        "official_target_expected": {
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.6,
            "dtype": "float16",
            "enforce_eager": False,
        },
        "historical_captured": "unknown/config-unverified",
        "note": (
            "official-target expected config; the config actually captured "
            "at original run time was not recorded, and a matched "
            "resolved_spec_hash does not prove config identity (issue #151)"
        ),
    },
    "client_config": {
        "protocol": "online serving",
        "workload_source": "standard benchmark workload",
    },
    "provenance": {
        "engine_repo": "vLLM-HUST/vllm-hust",
        "plugin_repo": "vLLM-HUST/vllm-ascend-hust",
        "base_backend_version": (
            "N/A (historical record, not captured at original run time)"
        ),
        "head_backend_version": (
            "N/A (historical record, not captured at original run time)"
        ),
        "note": (
            "Original leaderboard records have backend_version=N/A; retest "
            "captures full provenance via env-manifest.json"
        ),
    },
    "tracking_issue": "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/191",
    "related_prs": "#165 (methodology), #69/#77 (target commits)",
}

# Acceptance thresholds
TTFT_INCREASE_THRESHOLD = 0.20  # 20%
TPOT_INCREASE_THRESHOLD = 0.20  # 20%
THROUGHPUT_DECREASE_THRESHOLD = 0.10  # 10%
REPS_REQUIRED = 3  # exactly 3 valid reps per side (fail-closed)
MAX_FAILURE_RATE = 0.01  # reject a rep whose failure rate exceeds 1%

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

METRIC_FIELDS = ("mean_ttft_ms", "mean_tpot_ms", "output_throughput")


def log(msg: str) -> None:
    """Print a timestamped message to stderr."""
    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
    print(f"[{ts}] {msg}", file=sys.stderr, flush=True)


def is_valid_sha(sha: str | None) -> bool:
    """Check if a string is a valid 40-char hex SHA.

    Args:
        sha: The string to check.

    Returns:
        True if sha matches a 40-character lowercase hex string.
    """
    if not sha:
        return False
    return bool(SHA_RE.match(sha))


def verify_commit_in_repo(repo: str, sha: str) -> bool:
    """Verify that a commit SHA exists in the given git repo.

    Args:
        repo: Path to the git repository.
        sha: Full 40-char commit SHA to verify.

    Returns:
        True if the SHA resolves to a commit object in the repo.
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo, "cat-file", "-t", sha],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "commit" in result.stdout
    except (subprocess.SubprocessError, OSError):
        return False


def load_env_manifest(rep_dir: Path) -> dict[str, Any] | None:
    """Load and validate env-manifest.json from a rep directory.

    Args:
        rep_dir: Path to the rep-N directory.

    Returns:
        Parsed env-manifest dict, or None if missing/invalid.
    """
    manifest_file = rep_dir / "env-manifest.json"
    if not manifest_file.is_file():
        log(f"  WARNING: env-manifest.json missing in {rep_dir}")
        return None
    try:
        with open(manifest_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log(f"  WARNING: failed to parse {manifest_file}: {e}")
        return None


def validate_env_manifest(
    manifest: dict[str, Any],
    engine_repo: str | None,
    plugin_repo: str | None,
) -> bool:
    """Validate env-manifest fields for provenance integrity.

    Args:
        manifest: Parsed env-manifest dict.
        engine_repo: Optional path to engine repo for SHA verification.
        plugin_repo: Optional path to plugin repo for SHA verification.

    Returns:
        True if all validation checks pass, False otherwise.
    """
    engine_sha = manifest.get("engine_commit_observed", "")
    if not is_valid_sha(engine_sha):
        log(f"  FAIL: engine_commit_observed is not a valid SHA: {engine_sha!r}")
        return False

    if engine_repo:
        if not verify_commit_in_repo(engine_repo, engine_sha):
            log(f"  FAIL: engine SHA {engine_sha[:12]} not found in {engine_repo}")
            return False

    plugin_sha = manifest.get("plugin_commit_observed", "")
    if not is_valid_sha(plugin_sha):
        log(f"  FAIL: plugin_commit_observed is not a valid SHA: {plugin_sha!r}")
        return False

    if plugin_repo:
        if not verify_commit_in_repo(plugin_repo, plugin_sha):
            log(f"  FAIL: plugin SHA {plugin_sha[:12]} not found in {plugin_repo}")
            return False

    return True


def load_raw_metrics(rep_dir: Path) -> dict[str, Any] | None:
    """Load raw.json and extract benchmark metrics.

    Args:
        rep_dir: Path to the rep-N directory.

    Returns:
        Dict with mean_ttft_ms, mean_tpot_ms, output_throughput,
        or None if missing/invalid.
    """
    raw_file = rep_dir / "raw.json"
    if not raw_file.is_file():
        log(f"  WARNING: raw.json missing in {rep_dir}")
        return None
    try:
        with open(raw_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        log(f"  WARNING: failed to parse {raw_file}: {e}")
        return None

    metrics: dict[str, Any] = {}
    for field in METRIC_FIELDS:
        val = data.get(field)
        if val is None:
            log(f"  WARNING: {field} missing in {raw_file}")
            return None
        if not isinstance(val, (int, float)):
            log(f"  WARNING: {field} is not numeric in {raw_file}: {val!r}")
            return None
        metrics[field] = float(val)

    # Request-completion counters are REQUIRED (fail-closed): a rep that
    # does not report how many requests completed/failed cannot be
    # adjudicated and is rejected.
    for field in ("completed", "failed"):
        val = data.get(field)
        if val is None:
            log(f"  WARNING: {field} missing in {raw_file}")
            return None
        if not isinstance(val, (int, float)):
            log(f"  WARNING: {field} is not numeric in {raw_file}: {val!r}")
            return None
        metrics[field] = float(val)

    return metrics


def collect_rep_results(
    result_dir: Path,
    commit: str,
    workload: str,
    engine_repo: str | None,
    plugin_repo: str | None,
) -> list[dict[str, Any]]:
    """Collect valid results from all reps for a (commit, workload) pair.

    A rep is valid only if it has:
    1. A .completed marker file
    2. A valid env-manifest.json (with valid SHAs)
    3. A valid raw.json with required metrics

    Args:
        result_dir: Root results directory.
        commit: Engine commit short hash.
        workload: Workload name.
        engine_repo: Optional engine repo path for SHA verification.
        plugin_repo: Optional plugin repo path for SHA verification.

    Returns:
        List of dicts, each containing metrics and rep number.
    """
    commit_dir = result_dir / commit / workload
    if not commit_dir.is_dir():
        log(f"  No results directory for {commit} / {workload}")
        return []

    results: list[dict[str, Any]] = []
    rep_dirs = sorted(
        d for d in commit_dir.iterdir() if d.is_dir() and d.name.startswith("rep-")
    )

    for rep_dir in rep_dirs:
        rep_name = rep_dir.name
        completed_marker = rep_dir / ".completed"

        if not completed_marker.is_file():
            log(f"  SKIP {rep_name}: no .completed marker")
            continue

        manifest = load_env_manifest(rep_dir)
        if manifest is None:
            continue

        if not validate_env_manifest(manifest, engine_repo, plugin_repo):
            log(f"  SKIP {rep_name}: env-manifest validation failed")
            continue

        metrics = load_raw_metrics(rep_dir)
        if metrics is None:
            continue

        # Failure-rate policy (fail-closed): a rep whose failure rate exceeds
        # MAX_FAILURE_RATE is not representative of the workload and is
        # rejected, so it does not count toward the reps_required quota.
        attempted = metrics["completed"] + metrics["failed"]
        failed = metrics["failed"]
        if attempted > 0 and failed / attempted > MAX_FAILURE_RATE:
            log(
                f"  SKIP {rep_name}: failure rate {int(failed)}/"
                f"{int(attempted)} exceeds {MAX_FAILURE_RATE:.0%}"
            )
            continue

        rep_num = int(rep_name.removeprefix("rep-"))
        results.append(
            {
                "rep": rep_num,
                "metrics": metrics,
                "manifest": manifest,
            }
        )
        log(
            f"  OK {rep_name}: ttft={metrics['mean_ttft_ms']:.2f}ms "
            f"tpot={metrics['mean_tpot_ms']:.2f}ms "
            f"tput={metrics['output_throughput']:.2f}"
        )

    return results


def compute_median(
    results: list[dict[str, Any]],
    field: str,
) -> float | None:
    """Compute median of a metric field across reps.

    Args:
        results: List of rep result dicts.
        field: Metric field name (e.g. "mean_ttft_ms").

    Returns:
        Median value, or None if no valid results.
    """
    values = [r["metrics"][field] for r in results]
    if not values:
        return None
    return statistics.median(values)


def compare_interval(
    interval: dict[str, Any],
    result_dir: Path,
    engine_repo: str | None,
    plugin_repo: str | None,
) -> dict[str, Any]:
    """Compare base vs head for the issue #191 interval.

    Args:
        interval: Interval definition dict.
        result_dir: Root results directory.
        engine_repo: Optional engine repo path for SHA verification.
        plugin_repo: Optional plugin repo path for SHA verification.

    Returns:
        Dict with comparison results and verdict.
    """
    name = interval["name"]
    base_commit = interval["base_commit"]
    head_commit = interval["head_commit"]
    workload = interval["workload"]

    log(f"\n{'=' * 60}")
    log(f"Interval: {name}")
    log(f"  Base: {base_commit} ({interval['reported_jump']})")
    log(f"  Head: {head_commit}")
    log(f"{'=' * 60}")

    log(f"\nCollecting base results ({base_commit} / {workload}):")
    base_results = collect_rep_results(
        result_dir, base_commit, workload, engine_repo, plugin_repo
    )

    log(f"\nCollecting head results ({head_commit} / {workload}):")
    head_results = collect_rep_results(
        result_dir, head_commit, workload, engine_repo, plugin_repo
    )

    # Fail-closed: exactly REPS_REQUIRED valid reps are required on each
    # side. A side with fewer (or more) valid reps cannot be adjudicated.
    if len(base_results) != REPS_REQUIRED:
        log(
            f"  ERROR: base has {len(base_results)} valid reps, "
            f"expected exactly {REPS_REQUIRED}"
        )
    if len(head_results) != REPS_REQUIRED:
        log(
            f"  ERROR: head has {len(head_results)} valid reps, "
            f"expected exactly {REPS_REQUIRED}"
        )

    if len(base_results) != REPS_REQUIRED or len(head_results) != REPS_REQUIRED:
        return {
            "interval": name,
            "workload": workload,
            "base_commit": base_commit,
            "head_commit": head_commit,
            "reported_jump": interval["reported_jump"],
            "original_leaderboard": interval.get("original_leaderboard", {}),
            "absolute_value_drift": {
                "note": "",
                "caveat": (
                    "Retest and original leaderboard use different metric "
                    "definitions (mean_ttft_ms vs ttft_ms) under different "
                    "load profiles, so absolute values are not directly "
                    "comparable."
                ),
            },
            "base_reps": len(base_results),
            "head_reps": len(head_results),
            "reps_required": REPS_REQUIRED,
            "verdict": "incomplete_evidence",
            "reason": (
                f"Need exactly {REPS_REQUIRED} valid reps per side "
                f"(base={len(base_results)}, head={len(head_results)})"
            ),
        }

    # Compute medians
    base_ttft = compute_median(base_results, "mean_ttft_ms")
    head_ttft = compute_median(head_results, "mean_ttft_ms")
    base_tpot = compute_median(base_results, "mean_tpot_ms")
    head_tpot = compute_median(head_results, "mean_tpot_ms")
    base_tput = compute_median(base_results, "output_throughput")
    head_tput = compute_median(head_results, "output_throughput")

    assert base_ttft is not None
    assert head_ttft is not None
    assert base_tpot is not None
    assert head_tpot is not None
    assert base_tput is not None
    assert head_tput is not None

    # Compute relative changes
    ttft_change = (head_ttft - base_ttft) / base_ttft if base_ttft > 0 else 0
    tpot_change = (head_tpot - base_tpot) / base_tpot if base_tpot > 0 else 0
    tput_change = (head_tput - base_tput) / base_tput if base_tput > 0 else 0

    log(
        f"\n  Median TTFT:  base={base_ttft:.2f}ms  "
        f"head={head_ttft:.2f}ms  "
        f"change={ttft_change:+.1%}"
    )
    log(
        f"  Median TPOT:  base={base_tpot:.2f}ms  "
        f"head={head_tpot:.2f}ms  "
        f"change={tpot_change:+.1%}"
    )
    log(
        f"  Median Tput:  base={base_tput:.2f}    "
        f"head={head_tput:.2f}    "
        f"change={tput_change:+.1%}"
    )

    # Check thresholds
    ttft_regression = ttft_change > TTFT_INCREASE_THRESHOLD
    tpot_regression = tpot_change > TPOT_INCREASE_THRESHOLD
    tput_regression = tput_change < -THROUGHPUT_DECREASE_THRESHOLD

    any_regression = ttft_regression or tpot_regression or tput_regression

    if any_regression:
        reasons = []
        if ttft_regression:
            reasons.append(f"TTFT {ttft_change:+.1%} > {TTFT_INCREASE_THRESHOLD:.0%}")
        if tpot_regression:
            reasons.append(f"TPOT {tpot_change:+.1%} > {TPOT_INCREASE_THRESHOLD:.0%}")
        if tput_regression:
            reasons.append(
                f"Throughput {tput_change:+.1%} < -{THROUGHPUT_DECREASE_THRESHOLD:.0%}"
            )
        verdict = "reproducible_regression"
        reason = "; ".join(reasons)
    else:
        verdict = "not_reproducible"
        reason = (
            f"TTFT {ttft_change:+.1%}, TPOT {tpot_change:+.1%}, "
            f"Throughput {tput_change:+.1%} — all within thresholds"
        )

    log(f"\n  VERDICT: {verdict}")
    log(f"  REASON:  {reason}")

    return {
        "interval": name,
        "workload": workload,
        "base_commit": base_commit,
        "head_commit": head_commit,
        "reported_jump": interval["reported_jump"],
        "original_leaderboard": interval.get("original_leaderboard", {}),
        "absolute_value_drift": {
            "note": (
                "Retest and original leaderboard use different metric "
                "definitions (mean_ttft_ms vs ttft_ms) under different load "
                "profiles, so absolute values are not directly comparable."
            ),
            "caveat": (
                "Retest and original leaderboard use different metric "
                "definitions (mean_ttft_ms vs ttft_ms) under different load "
                "profiles, so absolute values are not directly comparable."
            ),
        },
        "base_reps": len(base_results),
        "head_reps": len(head_results),
        "reps_required": REPS_REQUIRED,
        "medians": {
            "base": {
                "mean_ttft_ms": base_ttft,
                "mean_tpot_ms": base_tpot,
                "output_throughput": base_tput,
            },
            "head": {
                "mean_ttft_ms": head_ttft,
                "mean_tpot_ms": head_tpot,
                "output_throughput": head_tput,
            },
        },
        "relative_changes": {
            "ttft": ttft_change,
            "tpot": tpot_change,
            "throughput": tput_change,
        },
        "thresholds": {
            "ttft_increase": TTFT_INCREASE_THRESHOLD,
            "tpot_increase": TPOT_INCREASE_THRESHOLD,
            "throughput_decrease": THROUGHPUT_DECREASE_THRESHOLD,
        },
        "completion_rate": {
            "requested_prompts_per_rep": 1000,
            "base": {
                "completed": int(sum(r["metrics"]["completed"] for r in base_results)),
                "failed": int(sum(r["metrics"]["failed"] for r in base_results)),
            },
            "head": {
                "completed": int(sum(r["metrics"]["completed"] for r in head_results)),
                "failed": int(sum(r["metrics"]["failed"] for r in head_results)),
            },
            "max_failure_rate": MAX_FAILURE_RATE,
            "note": (
                "Retest rerun without --enable-multimodal-chat: that flag "
                "pre-converted prompts into chat lists which were then wrapped "
                "again as a text field, so base64 images were tokenized as text "
                "and overflowed the 32768-token context (HTTP 400). Removing it "
                "keeps the official prompt structure (text + image_url), giving "
                "a completion rate well above the 1% failure-rate gate; reps "
                "failing the gate are rejected fail-closed and do not count "
                "toward the required 3 reps."
            ),
        },
        "verdict": verdict,
        "reason": reason,
    }


def main() -> int:
    """Main entry point for the analysis script.

    Returns:
        0 if analysis completed (regardless of verdict), 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Analyze issue #191 regression re-test results (visionarena-online)."
    )
    parser.add_argument(
        "--result-dir",
        default=DEFAULT_RESULT_DIR,
        help=f"Results directory (default: {DEFAULT_RESULT_DIR})",
    )
    parser.add_argument(
        "--engine-repo",
        default=None,
        help="Engine repo path for commit SHA verification (optional)",
    )
    parser.add_argument(
        "--plugin-repo",
        default=None,
        help="Plugin repo path for commit SHA verification (optional)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (default: print to stdout)",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        log(f"ERROR: result directory not found: {result_dir}")
        return 1

    log(f"Analyzing results in {result_dir}")
    log(f"Engine repo: {args.engine_repo or '(not provided)'}")
    log(f"Plugin repo: {args.plugin_repo or '(not provided)'}")

    interval_result = compare_interval(
        INTERVAL,
        result_dir,
        args.engine_repo,
        args.plugin_repo,
    )

    # Build final report
    report: dict[str, Any] = {
        "issue": "#191",
        "description": (
            "Regression re-test for the unexplained single-card leaderboard "
            "jump on visionarena-online (ec4847981f -> ceec19abb0)"
        ),
        "result_dir": str(result_dir),
        "interval": interval_result,
        "summary": {},
    }

    # Overall summary (single interval)
    verdict = interval_result["verdict"]
    report["summary"] = {
        "total_intervals": 1,
        "reproducible_regressions": 1 if verdict == "reproducible_regression" else 0,
        "not_reproducible": 1 if verdict == "not_reproducible" else 0,
        "incomplete_evidence": 1 if verdict == "incomplete_evidence" else 0,
        "overall_verdict": (
            "regression_reproduced"
            if verdict == "reproducible_regression"
            else "all_within_thresholds"
            if verdict == "not_reproducible"
            else "incomplete_evidence"
        ),
    }

    # Print summary
    log(f"\n{'=' * 60}")
    log("OVERALL SUMMARY")
    log(f"{'=' * 60}")
    status_icon = {
        "reproducible_regression": "[FAIL]",
        "not_reproducible": "[PASS]",
        "incomplete_evidence": "[????]",
    }.get(verdict, "[????]")
    log(
        f"  {status_icon} {interval_result['interval']} "
        f"({interval_result['base_commit']} -> {interval_result['head_commit']})"
    )
    log(f"  Verdict: {verdict}")
    log(f"  Reason:  {interval_result['reason']}")
    log(f"  Overall verdict: {report['summary']['overall_verdict']}")

    # Output JSON
    output_json = json.dumps(report, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_json)
        log(f"\nReport saved to {output_path}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())
