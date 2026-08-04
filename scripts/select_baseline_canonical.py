#!/usr/bin/env python3
"""Select canonical run from 3 baseline repetitions and update baseline submission.

Unlike select_canonical_repetition.py (which expects submission/ subdir in each rep),
this script loads the existing baseline run_leaderboard.json as template and only
updates metrics + adds measurement block from 3 raw_benchmark_result.json files.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

PRIMARY_METRIC = "throughput_tps"
PERFORMANCE_METRICS = ("throughput_tps", "ttft_ms", "tbt_ms")
SELECTED_RUN_METRICS = (
    "throughput_tps",
    "ttft_ms",
    "tbt_ms",
    "error_rate",
    "peak_mem_mb",
)
ERROR_RATE_TOLERANCE = 0.01


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_metrics(raw_path: Path) -> dict:
    from vllm_hust_benchmark.leaderboard_export import (
        _derive_metrics_from_benchmark_result,
    )

    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    derived = _derive_metrics_from_benchmark_result(payload, peak_mem_mb=None)
    metrics = {}
    for name in PERFORMANCE_METRICS:
        val = derived.get(name)
        if val is not None:
            val = float(val)
        metrics[name] = val
    err = derived.get("error_rate", 0)
    metrics["error_rate"] = float(err) if err is not None else 0.0
    peak = derived.get("peak_mem_mb")
    metrics["peak_mem_mb"] = float(peak) if peak is not None else None
    return metrics


def select_baseline_canonical(rep_dirs, baseline_submission_path, output_dir):
    """Select canonical run and update baseline submission with measurement block."""
    if len(rep_dirs) < 3:
        raise ValueError(f"Need 3 repetitions, got {len(rep_dirs)}")

    per_run = []
    for idx, rep_dir in enumerate(rep_dirs, start=1):
        raw_path = rep_dir / "raw_benchmark_result.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"No raw_benchmark_result.json in {rep_dir}")
        metrics = extract_metrics(raw_path)
        sha = sha256_file(raw_path)
        per_run.append(
            {
                "run_index": idx,
                "raw_result_sha256": sha,
                "metrics": metrics,
            }
        )
        err_str = (
            f"err={metrics['error_rate']:.4f}" if metrics["error_rate"] > 0 else "err=0"
        )
        print(
            f"  Rep {idx}: {PRIMARY_METRIC}={metrics[PRIMARY_METRIC]:.2f}, "
            f"ttft={metrics['ttft_ms']:.2f}, tbt={metrics['tbt_ms']:.2f}, {err_str}"
        )

    # Check error_rate tolerance
    for run in per_run:
        if run["metrics"]["error_rate"] > ERROR_RATE_TOLERANCE:
            print(
                f"  WARNING: Rep {run['run_index']} has error_rate="
                f"{run['metrics']['error_rate']:.4f} (>{ERROR_RATE_TOLERANCE}), excluding"
            )
            run["excluded"] = True

    valid_runs = [r for r in per_run if not r.get("excluded")]
    if len(valid_runs) < 1:
        raise ValueError("All runs have error_rate > tolerance")

    # Select primary-median-run: sort by primary metric, pick middle
    ordered = sorted(
        valid_runs, key=lambda r: (r["metrics"][PRIMARY_METRIC], r["run_index"])
    )
    selected_position = len(ordered) // 2 + 1
    selected = ordered[selected_position - 1]

    print(
        f"\n  Selected: rep {selected['run_index']} (position {selected_position}/{len(ordered)})"
    )
    print(
        f"  Canonical metrics: ttft={selected['metrics']['ttft_ms']:.2f}, "
        f"tbt={selected['metrics']['tbt_ms']:.2f}, "
        f"throughput={selected['metrics'][PRIMARY_METRIC]:.2f}"
    )

    # Build measurement block
    selected_metrics = {
        name: float(selected["metrics"][name])
        for name in SELECTED_RUN_METRICS
        if selected["metrics"].get(name) is not None
    }

    measurement = {
        "schema_version": "perfgate-measurement/v2",
        "strategy": "warmup+primary-median-run",
        "warmup_runs": 0,
        "measured_runs": len(per_run),
        "aggregation": "primary-median-run",
        "selection": {
            "primary_metric": PRIMARY_METRIC,
            "sort_direction": "ascending",
            "secondary_sort_key": "run_index",
            "ordered_run_indices": [r["run_index"] for r in ordered],
            "selected_position": selected_position,
            "selected_run_index": selected["run_index"],
            "selected_raw_result_sha256": selected["raw_result_sha256"],
        },
        "warmup": [],
        "per_run": [
            {
                "run_index": r["run_index"],
                "raw_result_sha256": r["raw_result_sha256"],
                "metrics": {
                    name: (
                        float(r["metrics"][name])
                        if r["metrics"].get(name) is not None
                        else None
                    )
                    for name in (
                        "throughput_tps",
                        "ttft_ms",
                        "tbt_ms",
                        "error_rate",
                        "peak_mem_mb",
                    )
                },
            }
            for r in per_run
        ],
    }

    # Load existing baseline run_leaderboard.json as template
    baseline_path = Path(baseline_submission_path)
    canonical = json.loads(baseline_path.read_text(encoding="utf-8"))
    canonical["measurement"] = measurement
    canonical["metrics"] = selected_metrics

    # Update metadata to reflect 3 repetitions
    if "metadata" in canonical:
        canonical["metadata"]["repetitions"] = len(per_run)
        canonical["metadata"]["measurement_strategy"] = "warmup+primary-median-run"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "run_leaderboard.json"
    out_path.write_text(json.dumps(canonical, indent=2, ensure_ascii=False))

    print(f"\n  Written canonical baseline submission to {out_path}")
    return canonical


def main():
    parser = argparse.ArgumentParser(
        description="Select canonical baseline run from 3 repetitions"
    )
    parser.add_argument("--rep1", required=True, help="Rep 1 directory")
    parser.add_argument("--rep2", required=True, help="Rep 2 directory")
    parser.add_argument("--rep3", required=True, help="Rep 3 directory")
    parser.add_argument(
        "--baseline", required=True, help="Existing baseline run_leaderboard.json path"
    )
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    rep_dirs = [Path(d) for d in [args.rep1, args.rep2, args.rep3]]
    select_baseline_canonical(rep_dirs, args.baseline, args.output)


if __name__ == "__main__":
    main()
