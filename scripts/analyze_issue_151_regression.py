#!/usr/bin/env python3
"""Analyze issue #151 regression re-test results.

Reads raw.json benchmark results from the retest_issue_151_regression.sh
output directory, computes median metrics across interleaved repetitions,
and compares base vs head commits for each interval to determine if the
reported performance jumps are reproducible regressions or noise.

Usage:
    python analyze_issue_151_regression.py \\
        --result-dir /data/issue151-retest-results \\
        [--engine-repo /root/vllm/vllm-hust] \\
        [--plugin-repo /root/vllm/vllm-ascend-hust]

Acceptance thresholds (from issue #151):
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RESULT_DIR = str(
    Path(__file__).resolve().parents[1] / "reports" / "issue_151_retest_raw_results"
)

# Interval definitions: (name, base_commit, head_commit, workload)
INTERVALS = [
    {
        "name": "random-online",
        "base_commit": "2206f1f7b7",
        "head_commit": "f273f9c5e2",
        "workload": "random-online",
        "reported_jump": "173% TTFT jump",
        "original_leaderboard": {
            "base": {"ttft_ms": 302.776655042544, "throughput_tps": 238.4099656585595},
            "head": {
                "ttft_ms": 2243.397351723397,
                "throughput_tps": 236.86996143257514,
            },
        },
        "absolute_value_drift_note": (
            "Retest mean_ttft_ms (~131s base, ~128s head) is far above the original "
            "leaderboard ttft_ms (~0.3s base, ~2.2s head). The retest did NOT reproduce "
            "the original absolute magnitude; the original jump was a single-run outlier "
            "and the retest environment operates at a different absolute load profile. "
            "Conclusion is 'original data noise' rather than a reproducible regression."
        ),
    },
    {
        "name": "agent-research-online",
        "base_commit": "7a63f81e86",
        "head_commit": "ec4847981f",
        "workload": "agent-research-online",
        "reported_jump": "7.8x TTFT jump",
        "original_leaderboard": {
            "base": {
                "ttft_ms": 330.0492724083597,
                "throughput_tps": 184.71806876637058,
            },
            "head": {
                "ttft_ms": 3443.249249634391,
                "throughput_tps": 138.39660869279373,
            },
        },
        "absolute_value_drift_note": (
            "Retest mean_ttft_ms (~2.3s base, ~2.3s head) sits between the original "
            "leaderboard ttft_ms (~0.3s base, ~3.4s head). The original 7.8x jump was "
            "driven by the head single-run outlier; the retest does not reproduce it, "
            "placing the conclusion as 'original data noise' rather than a reproducible "
            "regression."
        ),
    },
]

# Metric definitions shared across all remaining intervals.
METRIC_DEFINITIONS = {
    "ttft_ms": "Time To First Token in milliseconds (lower is better)",
    "tpot_ms": "Time Per Output Token in milliseconds (lower is better)",
    "throughput_tps": "Output throughput in tokens per second (higher is better)",
}

# Remaining intervals from issue #151 that PR #165 did not re-test. Tracked by
# issue #177. Each entry carries the full provenance contract but marks
# ``retest_status="pending"`` because no interleaved retest has run yet, so the
# verdict is ``incomplete_evidence`` with disposition ``rerun``. The original
# leaderboard values below are taken verbatim from the issue #151 table; backend
# version provenance was not captured at original run time.
REMAINING_INTERVALS = [
    {
        "interval_id": "agent-research-online-f273f9c5e2-51621c35bc",
        "workload": "agent-research-online",
        "base_commit": "f273f9c5e2",
        "head_commit": "51621c35bc",
        "retest_status": "pending",
        "reported_jump": {
            "ttft_ms": {"base": 281, "head": 434, "change_pct": 54.2},
            "tpot_ms": {"base": 49.1, "head": 55.7, "change_pct": 13.4},
            "throughput_tps": {"base": 187.9, "head": 180.8, "change_pct": -3.7},
        },
        "original_leaderboard": {
            "base": {"ttft_ms": 281, "tpot_ms": 49.1, "throughput_tps": 187.9},
            "head": {"ttft_ms": 434, "tpot_ms": 55.7, "throughput_tps": 180.8},
        },
        "hardware": {"chip_model": "910B2", "chip_count": 1, "node_count": 1},
        "model": {"name": "Qwen2.5-14B-Instruct", "precision": "float16"},
        "same_spec_identity": (
            "resolved_spec_hash matched between base and head (per issue #151); "
            "spec_id=official-ascend-jan-2026-v0.18.0-"
            "agent-research-online-qwen25-14b-910b2"
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
                "required to capture full provenance"
            ),
        },
        "reps_completed": 0,
        "reps_required": 3,
        "verdict": "incomplete_evidence",
        "disposition": "rerun",
        "disposition_reason": (
            "No retest performed yet; original records lack backend version "
            "provenance. Must rerun with 3 interleaved reps per side using the "
            "same metric/config contract as #165 before concluding."
        ),
        "tracking_issue": "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/188",
        "related_prs": "#165 (methodology), #49/#41 (target commits)",
    },
    {
        "interval_id": "instructcoder-online-51621c35bc-7a63f81e86",
        "workload": "instructcoder-online",
        "base_commit": "51621c35bc",
        "head_commit": "7a63f81e86",
        "retest_status": "pending",
        "reported_jump": {
            "ttft_ms": {"base": 244, "head": 297, "change_pct": 21.4},
            "tpot_ms": {"base": 49.8, "head": 54.7, "change_pct": 9.9},
            "throughput_tps": {"base": 167.3, "head": 167.7, "change_pct": 0.2},
        },
        "original_leaderboard": {
            "base": {"ttft_ms": 244, "tpot_ms": 49.8, "throughput_tps": 167.3},
            "head": {"ttft_ms": 297, "tpot_ms": 54.7, "throughput_tps": 167.7},
        },
        "hardware": {"chip_model": "910B2", "chip_count": 1, "node_count": 1},
        "model": {"name": "Qwen2.5-Coder-14B-Instruct", "precision": "float16"},
        "same_spec_identity": (
            "resolved_spec_hash matched between base and head (per issue #151); "
            "spec_id=official-ascend-jan-2026-v0.18.0-"
            "instructcoder-online-qwen25-coder-14b-910b2"
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
                "required to capture full provenance"
            ),
        },
        "reps_completed": 0,
        "reps_required": 3,
        "verdict": "incomplete_evidence",
        "disposition": "rerun",
        "disposition_reason": (
            "No retest performed yet; original records lack backend version "
            "provenance. Must rerun with 3 interleaved reps per side using the "
            "same metric/config contract as #165 before concluding."
        ),
        "tracking_issue": "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/189",
        "related_prs": "#165 (methodology), #41/#66 (target commits)",
    },
    {
        "interval_id": "random-online-7a63f81e86-ec4847981f",
        "workload": "random-online",
        "base_commit": "7a63f81e86",
        "head_commit": "ec4847981f",
        "retest_status": "pending",
        "reported_jump": {
            "ttft_ms": {"base": 1261, "head": 1535, "change_pct": 21.7},
            "tpot_ms": {"base": 52.1, "head": 54.0, "change_pct": 3.6},
            "throughput_tps": {"base": 238.9, "head": 237.4, "change_pct": -0.7},
        },
        "original_leaderboard": {
            "base": {"ttft_ms": 1261, "tpot_ms": 52.1, "throughput_tps": 238.9},
            "head": {"ttft_ms": 1535, "tpot_ms": 54.0, "throughput_tps": 237.4},
        },
        "hardware": {"chip_model": "910B2", "chip_count": 1, "node_count": 1},
        "model": {"name": "Qwen2.5-14B-Instruct", "precision": "float16"},
        "same_spec_identity": (
            "resolved_spec_hash matched between base and head (per issue #151); "
            "spec_id=official-ascend-jan-2026-v0.18.0-"
            "random-online-qwen25-14b-910b2"
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
                "required to capture full provenance"
            ),
        },
        "reps_completed": 0,
        "reps_required": 3,
        "verdict": "incomplete_evidence",
        "disposition": "rerun",
        "disposition_reason": (
            "No retest performed yet; original records lack backend version "
            "provenance. Must rerun with 3 interleaved reps per side using the "
            "same metric/config contract as #165 before concluding."
        ),
        "tracking_issue": "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/190",
        "related_prs": "#165 (methodology), #66/#69 (target commits)",
    },
    {
        "interval_id": "visionarena-online-ec4847981f-ceec19abb0",
        "workload": "visionarena-online",
        "base_commit": "ec4847981f",
        "head_commit": "ceec19abb0",
        "retest_status": "pending",
        "reported_jump": {
            "ttft_ms": {"base": 369, "head": 387, "change_pct": 5.0},
            "tpot_ms": {"base": 35.2, "head": 60.3, "change_pct": 71.4},
            "throughput_tps": {"base": 117.0, "head": 118.2, "change_pct": 1.0},
        },
        "original_leaderboard": {
            "base": {"ttft_ms": 369, "tpot_ms": 35.2, "throughput_tps": 117.0},
            "head": {"ttft_ms": 387, "tpot_ms": 60.3, "throughput_tps": 118.2},
        },
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
                "required to capture full provenance"
            ),
        },
        "reps_completed": 0,
        "reps_required": 3,
        "verdict": "incomplete_evidence",
        "disposition": "rerun",
        "disposition_reason": (
            "No retest performed yet; original records lack backend version "
            "provenance. Must rerun with 3 interleaved reps per side using the "
            "same metric/config contract as #165 before concluding."
        ),
        "tracking_issue": "https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/191",
        "related_prs": "#165 (methodology), #69/#77 (target commits)",
    },
]

# Acceptance thresholds
TTFT_INCREASE_THRESHOLD = 0.20  # 20%
TPOT_INCREASE_THRESHOLD = 0.20  # 20%
THROUGHPUT_DECREASE_THRESHOLD = 0.10  # 10%

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
    interval: dict[str, str],
    result_dir: Path,
    engine_repo: str | None,
    plugin_repo: str | None,
) -> dict[str, Any]:
    """Compare base vs head for a single interval.

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

    # Need at least 1 valid rep on each side
    if not base_results:
        log(f"  ERROR: no valid base results for {name}")
    if not head_results:
        log(f"  ERROR: no valid head results for {name}")

    if not base_results or not head_results:
        return {
            "interval": name,
            "workload": workload,
            "base_commit": base_commit,
            "head_commit": head_commit,
            "reported_jump": interval["reported_jump"],
            "original_leaderboard": interval.get("original_leaderboard", {}),
            "absolute_value_drift": {
                "note": interval.get("absolute_value_drift_note", ""),
                "caveat": (
                    "Retest and original leaderboard use different metric definitions "
                    "(mean_ttft_ms vs ttft_ms) under different load profiles, so "
                    "absolute values are not directly comparable."
                ),
            },
            "base_reps": len(base_results),
            "head_reps": len(head_results),
            "verdict": "incomplete_evidence",
            "reason": "Insufficient valid reps on one or both sides",
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
        if interval.get("absolute_value_drift_note"):
            reason += ". Absolute values did not reproduce original magnitude (see absolute_value_drift)"

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
            "note": interval.get("absolute_value_drift_note", ""),
            "caveat": (
                "Retest and original leaderboard use different metric definitions "
                "(mean_ttft_ms vs ttft_ms) under different load profiles, so absolute "
                "values are not directly comparable."
            ),
        },
        "base_reps": len(base_results),
        "head_reps": len(head_results),
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
        "verdict": verdict,
        "reason": reason,
    }


def generate_remaining_jumps_report(
    output_path: str | None = None,
) -> dict[str, Any]:
    """Build the issue #177 machine-readable report for the remaining jumps.

    Assembles the tracking report for the 4 unexplained single-card leaderboard
    jumps that PR #165 did not re-test. Every interval is marked
    ``incomplete_evidence`` with disposition ``rerun`` because no interleaved
    retest has been performed yet and the original records lack backend-version
    provenance.

    Args:
        output_path: Optional path to write the JSON report to. When omitted the
            assembled report dict is returned without writing to disk.

    Returns:
        The assembled issue #177 report dict.
    """
    intervals: list[dict[str, Any]] = []
    for iv in REMAINING_INTERVALS:
        intervals.append(
            {
                "interval_id": iv["interval_id"],
                "workload": iv["workload"],
                "base_commit": iv["base_commit"],
                "head_commit": iv["head_commit"],
                "retest_status": iv["retest_status"],
                "reported_jump": iv["reported_jump"],
                "original_leaderboard": iv["original_leaderboard"],
                "metric_definitions": dict(METRIC_DEFINITIONS),
                "hardware": dict(iv["hardware"]),
                "model": dict(iv["model"]),
                "same_spec_identity": iv["same_spec_identity"],
                "server_config": dict(iv["server_config"]),
                "client_config": dict(iv["client_config"]),
                "provenance": dict(iv["provenance"]),
                "reps_completed": iv["reps_completed"],
                "reps_required": iv["reps_required"],
                "verdict": iv["verdict"],
                "disposition": iv["disposition"],
                "disposition_reason": iv["disposition_reason"],
                "tracking_issue": iv["tracking_issue"],
                "related_prs": iv["related_prs"],
            }
        )

    report: dict[str, Any] = {
        "issue": "#177",
        "follow_up_for": "#165",
        "parent_issue": "vLLM-HUST/vllm-hust#151",
        "description": (
            "Tracking report for the 4 remaining unexplained single-card "
            "leaderboard jumps not covered by PR #165"
        ),
        "intervals": intervals,
        "summary": {
            "total_remaining_intervals": len(intervals),
            "reproducible_regressions": 0,
            "not_reproducible": 0,
            "incomplete_evidence": len(intervals),
            "overall_verdict": "incomplete_evidence",
            "disposition_summary": {
                "retain": 0,
                "quarantine": 0,
                "supersede": 0,
                "rerun": len(intervals),
            },
            "metric_config_contract_note": (
                "All 4 remaining intervals must use the same metric/config "
                "contract as #165: median-based comparison, 3 interleaved reps "
                "per side, fixed NPU/model/CANN/torch_npu/dtype/graph mode/"
                "concurrency/RPS. Original mean_ttft_ms vs original ttft_ms "
                "are not directly comparable (per #165 report)."
            ),
        },
    }

    if output_path:
        out = Path(output_path)
        out.write_text(json.dumps(report, indent=2) + "\n")
        log(f"Issue #177 remaining-jumps report saved to {out}")

    return report


def main() -> int:
    """Main entry point for the analysis script.

    Returns:
        0 if analysis completed (regardless of verdict), 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Analyze issue #151 regression re-test results."
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
    parser.add_argument(
        "--remaining-report",
        default=None,
        help=(
            "Generate the issue #177 remaining-jumps machine-readable report "
            "to the given path and exit (no retest data required)."
        ),
    )
    args = parser.parse_args()

    if args.remaining_report:
        generate_remaining_jumps_report(args.remaining_report)
        return 0

    result_dir = Path(args.result_dir)
    if not result_dir.is_dir():
        log(f"ERROR: result directory not found: {result_dir}")
        return 1

    log(f"Analyzing results in {result_dir}")
    log(f"Engine repo: {args.engine_repo or '(not provided)'}")
    log(f"Plugin repo: {args.plugin_repo or '(not provided)'}")

    interval_results = []
    for interval in INTERVALS:
        result = compare_interval(
            interval,
            result_dir,
            args.engine_repo,
            args.plugin_repo,
        )
        interval_results.append(result)

    # Build final report
    report: dict[str, Any] = {
        "issue": "#151",
        "description": (
            "Regression re-test for unexplained performance jumps in "
            "single-card leaderboard"
        ),
        "result_dir": str(
            result_dir.resolve().relative_to(Path(__file__).resolve().parents[1])
        )
        if result_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1])
        else str(result_dir),
        "intervals": interval_results,
        "summary": {},
    }

    # Overall summary
    reproducible = [
        r for r in interval_results if r["verdict"] == "reproducible_regression"
    ]
    not_reproducible = [
        r for r in interval_results if r["verdict"] == "not_reproducible"
    ]
    incomplete = [r for r in interval_results if r["verdict"] == "incomplete_evidence"]

    report["summary"] = {
        "total_intervals": len(interval_results),
        "reproducible_regressions": len(reproducible),
        "not_reproducible": len(not_reproducible),
        "incomplete_evidence": len(incomplete),
        "overall_verdict": (
            "regression_reproduced"
            if reproducible
            else "all_within_thresholds"
            if not_reproducible and not incomplete
            else "incomplete_evidence"
        ),
    }

    # Print summary
    log(f"\n{'=' * 60}")
    log("OVERALL SUMMARY")
    log(f"{'=' * 60}")
    log(f"  Total intervals:       {len(interval_results)}")
    log(f"  Reproducible:         {len(reproducible)}")
    log(f"  Not reproducible:     {len(not_reproducible)}")
    log(f"  Incomplete evidence:  {len(incomplete)}")
    log(f"  Overall verdict:      {report['summary']['overall_verdict']}")

    for r in interval_results:
        status_icon = {
            "reproducible_regression": "[FAIL]",
            "not_reproducible": "[PASS]",
            "incomplete_evidence": "[????]",
        }.get(r["verdict"], "[????]")
        log(
            f"  {status_icon} {r['interval']:30s} "
            f"({r['base_commit']} -> {r['head_commit']})"
        )

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
