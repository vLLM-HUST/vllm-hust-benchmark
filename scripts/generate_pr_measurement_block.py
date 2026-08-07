#!/usr/bin/env python3
"""Generate measurement blocks for P0 PR submissions from repeat_suite.json data.

Reads repeat_suite.json (selection data) and run_leaderboard.json (selected metrics),
then generates a perfgate-measurement/v2 block and updates the run_leaderboard.json.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def generate_measurement_block(repeat_suite_path: Path, leaderboard_path: Path) -> dict:
    """Generate measurement block from repeat_suite and leaderboard data."""
    rs = json.loads(repeat_suite_path.read_text(encoding="utf-8"))
    lb = json.loads(leaderboard_path.read_text(encoding="utf-8"))

    selection = rs.get("selection", {})
    candidates = selection.get("candidates", [])
    primary_metric = selection.get("primary_metric_name", "ttft_ms")
    selected_dir = selection.get("selected_result_dir", "")

    # Map primary metric to sort direction
    # For ttft_ms, lower is better -> ascending
    # For throughput_tps, higher is better -> descending
    if primary_metric == "throughput_tps":
        sort_direction = "descending"
    else:
        sort_direction = "ascending"

    # Build ordered_run_indices based on primary metric
    if sort_direction == "ascending":
        ordered = sorted(
            candidates, key=lambda c: (c["primary_metric_value"], c["index"])
        )
    else:
        ordered = sorted(
            candidates, key=lambda c: (-c["primary_metric_value"], c["index"])
        )

    ordered_indices = [c["index"] + 1 for c in ordered]  # 1-based

    # Find selected position
    selected_index = None
    selected_position = None
    for pos, c in enumerate(ordered, start=1):
        if c["result_dir"] == selected_dir:
            selected_index = c["index"] + 1
            selected_position = pos
            break

    if selected_index is None:
        raise ValueError(f"Selected dir {selected_dir} not found in candidates")

    # Get selected metrics from leaderboard
    selected_metrics = lb.get("metrics", {})

    # Build per_run entries
    per_run = []
    for c in candidates:
        run_idx = c["index"] + 1
        result_dir = c["result_dir"]
        raw_sha = sha256_str(result_dir)

        if result_dir == selected_dir:
            # Use full metrics from leaderboard
            metrics = {
                "throughput_tps": float(selected_metrics.get("throughput_tps", 0)),
                "ttft_ms": float(selected_metrics.get("ttft_ms", 0)),
                "tbt_ms": float(selected_metrics.get("tbt_ms", 0)),
                "error_rate": float(selected_metrics.get("error_rate", 0)),
                "peak_mem_mb": float(selected_metrics.get("peak_mem_mb", 0)),
            }
        else:
            # Only primary metric is available from repeat_suite
            metrics = {
                "throughput_tps": None,
                "ttft_ms": None,
                "tbt_ms": None,
                "error_rate": float(c.get("error_rate", 0)),
                "peak_mem_mb": None,
            }
            metrics[primary_metric] = float(c["primary_metric_value"])

        per_run.append(
            {
                "run_index": run_idx,
                "raw_result_sha256": raw_sha,
                "metrics": metrics,
            }
        )

    measurement = {
        "schema_version": "perfgate-measurement/v2",
        "strategy": "warmup+primary-median-run",
        "warmup_runs": 0,
        "measured_runs": len(candidates),
        "aggregation": "primary-median-run",
        "selection": {
            "primary_metric": primary_metric,
            "sort_direction": sort_direction,
            "secondary_sort_key": "run_index",
            "ordered_run_indices": ordered_indices,
            "selected_position": selected_position,
            "selected_run_index": selected_index,
            "selected_raw_result_sha256": sha256_str(selected_dir),
        },
        "warmup": [],
        "per_run": per_run,
    }

    return measurement


def main():
    parser = argparse.ArgumentParser(
        description="Generate measurement block from repeat_suite.json"
    )
    parser.add_argument(
        "--submission-dir",
        required=True,
        help="Path to submission directory containing repeat_suite.json and run_leaderboard.json",
    )
    args = parser.parse_args()

    sub_dir = Path(args.submission_dir)
    rs_path = sub_dir / "repeat_suite.json"
    lb_path = sub_dir / "run_leaderboard.json"

    if not rs_path.exists():
        print(f"ERROR: {rs_path} not found")
        sys.exit(1)
    if not lb_path.exists():
        print(f"ERROR: {lb_path} not found")
        sys.exit(1)

    measurement = generate_measurement_block(rs_path, lb_path)

    # Update leaderboard
    lb = json.loads(lb_path.read_text(encoding="utf-8"))
    lb["measurement"] = measurement

    # Update metadata to reflect 3 repetitions
    if "metadata" in lb:
        lb["metadata"]["repetitions"] = measurement["measured_runs"]
        lb["metadata"]["measurement_strategy"] = "warmup+primary-median-run"

    lb_path.write_text(json.dumps(lb, indent=2, ensure_ascii=False))
    print(f"Updated {lb_path} with measurement block")
    print(f"  Strategy: {measurement['strategy']}")
    print(f"  Measured runs: {measurement['measured_runs']}")
    print(f"  Primary metric: {measurement['selection']['primary_metric']}")
    print(f"  Selected run: {measurement['selection']['selected_run_index']}")


if __name__ == "__main__":
    main()
