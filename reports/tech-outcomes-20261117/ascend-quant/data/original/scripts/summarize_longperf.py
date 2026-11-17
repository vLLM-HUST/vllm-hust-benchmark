#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize vllm bench serve JSON files from run_longctx_perf_card7.sh."""
import argparse
import csv
import glob
import json
import os
import re
import statistics

FIELDS = [
    "completed", "failed", "duration",
    "total_input_tokens", "total_output_tokens",
    "request_throughput", "output_throughput", "total_token_throughput",
    "p50_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
    "p50_tpot_ms", "p90_tpot_ms", "p99_tpot_ms",
    "p50_itl_ms", "p90_itl_ms", "p99_itl_ms",
    "p50_e2el_ms", "p90_e2el_ms", "p99_e2el_ms",
]

NAME_RE = re.compile(r"^(w8a8|w4a8)_L(\d+)_r(\d+)\.json$")


def load_runs(dirpath):
    runs = []
    for path in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        arm, L, rep = m.group(1), int(m.group(2)), int(m.group(3))
        with open(path) as fh:
            d = json.load(fh)
        row = {"arm": arm, "length": L, "rep": rep, "file": os.path.basename(path)}
        for k in FIELDS:
            row[k] = d.get(k)
        runs.append(row)
    return runs


def med(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    runs = load_runs(args.dir)
    if not runs:
        raise SystemExit("no matching *_L*_r*.json in %s" % args.dir)

    runs_csv = os.path.join(args.dir, "longperf_runs.csv")
    with open(runs_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "length", "rep"] + FIELDS + ["file"])
        w.writeheader()
        for r in sorted(runs, key=lambda x: (x["arm"], x["length"], x["rep"])):
            w.writerow(r)

    groups = {}
    for r in runs:
        groups.setdefault((r["arm"], r["length"]), []).append(r)

    summary_csv = os.path.join(args.dir, "longperf_summary.csv")
    keys = ["p50_ttft_ms", "p90_ttft_ms", "p99_ttft_ms",
            "p50_tpot_ms", "p90_tpot_ms", "p50_itl_ms", "p90_itl_ms",
            "output_throughput", "total_token_throughput"]
    with open(summary_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "length", "reps", "completed", "failed",
                    "total_input_tokens", "total_output_tokens"] + keys)
        for (arm, L), rows in sorted(groups.items()):
            w.writerow([arm, L, len(rows),
                        med(rows, "completed"), med(rows, "failed"),
                        med(rows, "total_input_tokens"), med(rows, "total_output_tokens")]
                       + [med(rows, k) for k in keys])

    print("\narm     L        reps  TTFT p50/p90 ms   TPOT p50/p90 ms   out tok/s   total tok/s")
    for (arm, L), rows in sorted(groups.items()):
        print("  %-6s %-8d %-4d  %8.1f / %-8.1f  %7.1f / %-7.1f  %9.1f   %10.1f" % (
            arm, L, len(rows),
            med(rows, "p50_ttft_ms") or 0, med(rows, "p90_ttft_ms") or 0,
            med(rows, "p50_tpot_ms") or 0, med(rows, "p90_tpot_ms") or 0,
            med(rows, "output_throughput") or 0,
            med(rows, "total_token_throughput") or 0))
    print("\nwrote", runs_csv)
    print("wrote", summary_csv)


if __name__ == "__main__":
    main()
