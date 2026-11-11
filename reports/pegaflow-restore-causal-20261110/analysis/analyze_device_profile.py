"""Align device memcpy intervals with measured native or C control operations."""

import argparse
import csv
from decimal import Decimal
import json
from pathlib import Path


def analyze(intervals):
    ordered = sorted(intervals)
    if not ordered:
        return dict(memcpy_tasks=0)
    union = 0
    gaps = []
    left, right = ordered[0]
    total = sum(end - start for start, end in ordered)
    for start, end in ordered[1:]:
        if start > right:
            union += right - left
            gaps.append(start - right)
            left, right = start, end
        else:
            right = max(right, end)
    union += right - left
    return dict(
        memcpy_tasks=len(ordered),
        dma_union_ms=union / 1e6,
        dma_sum_ms=total / 1e6,
        device_span_ms=(right - ordered[0][0]) / 1e6,
        max_gap_ms=max(gaps, default=0) / 1e6,
        gaps_over_1ms=sum(gap > 1_000_000 for gap in gaps),
        last_dma_ns=right,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operations", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    intervals = []
    with args.csv.open() as stream:
        for row in csv.DictReader(stream):
            if row["kernel_type"] == "MEMCPY_ASYNC":
                intervals.append(
                    tuple(
                        int(Decimal(row[k].strip()) * 1000)
                        for k in ["task_start(us)", "task_stop(us)"]
                    )
                )
    operations = []
    for line in args.operations.read_text().splitlines():
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        if "complete_s" in row:
            start = row["start_epoch_ns"]
            elapsed = round(row["complete_s"] * 1e9)
        elif "wall_ns" in row:
            start, elapsed = row["epoch_ns"], row["wall_ns"]
        else:
            continue
        end = start + elapsed
        selected = [(a, b) for a, b in intervals if start <= a < end]
        result = analyze(selected)
        last = result.pop("last_dma_ns", None)
        result.update(
            phase=row.get("phase", row.get("repeat")),
            expected_memcpy_tasks=row["descriptors"],
            complete_task_coverage=len(selected) == row["descriptors"],
            wall_ms=elapsed / 1e6,
            tail_after_last_dma_ms=(end - last) / 1e6 if last else None,
        )
        if "api_ns" in row:
            result.update(
                api_ms=row["api_ns"] / 1e6,
                sync_ms=row["sync_ns"] / 1e6,
                caller_cpu_ms=row["cpu_ns"] / 1e6,
            )
        operations.append(result)
    args.output.write_text(
        json.dumps(
            dict(
                csv=str(args.csv),
                total_memcpy_rows=len(intervals),
                operations=operations,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(operations))


if __name__ == "__main__":
    main()
