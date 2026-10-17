#!/usr/bin/env python3
"""One table over every run in an experiment directory.

    python experiments/workload/compare_runs.py data/results/workload/<experiment>

Reads each <run>/summary.json written by replay_requests.py; prints one row per
run with the arrival parameters of its workload next to the measured latencies.
Only runs inside one directory are compared (same engine, same session).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLUMNS = (
    ("run", 38),
    ("arrival CV", 10),
    ("req/s", 6),
    ("ok/total", 9),
    ("TTFT p50", 9),
    ("TTFT p99", 9),
    ("TPOT p50", 9),
    ("TPOT p99", 9),
    ("E2E p50", 8),
    ("E2E p99", 8),
    ("out tok/s", 9),
    ("len mismatch", 12),
)


def row(run: Path) -> list[str]:
    summary = json.loads((run / "summary.json").read_text())
    result = summary["result"]
    arrival = (summary.get("workload_manifest") or {}).get("parameters", {})
    arrival = arrival.get("arrival", {})
    ttft = result["time_to_first_token_seconds"]
    tpot = result["time_per_output_token_seconds"]
    end_to_end = result["end_to_end_latency_seconds"]
    mismatches = (
        result["input_length_mismatches"]["count"]
        + result["output_length_mismatches"]["count"]
    )
    return [
        run.name,
        str(arrival.get("coefficient_of_variation", "?")),
        str(arrival.get("request_rate_per_second", "?")),
        f"{result['succeeded']}/{result['requests']}",
        f"{ttft.get('p50', 0):.3f}",
        f"{ttft.get('p99', 0):.3f}",
        f"{tpot.get('p50', 0):.4f}",
        f"{tpot.get('p99', 0):.4f}",
        f"{end_to_end.get('p50', 0):.1f}",
        f"{end_to_end.get('p99', 0):.1f}",
        f"{result['output_tokens_per_second']:.0f}",
        str(mismatches),
    ]


def main() -> None:
    experiment = Path(sys.argv[1])
    runs = sorted(path.parent for path in experiment.glob("*/summary.json"))
    print("  ".join(name.ljust(width) for name, width in COLUMNS))
    for run in runs:
        cells = row(run)
        print(
            "  ".join(
                cell.ljust(width)
                for cell, (_, width) in zip(cells, COLUMNS, strict=True)
            )
        )
    print("(seconds; TTFT = first token - sent, TPOT = decode time / (output - 1),")
    print(" E2E = finished - sent; definitions as vllm bench serve)")


if __name__ == "__main__":
    main()
