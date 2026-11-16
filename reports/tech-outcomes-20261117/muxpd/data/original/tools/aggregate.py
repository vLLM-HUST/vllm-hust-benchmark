"""Aggregate repeated passes: per plan and arm, mean and [min, max] across passes.

Usage: aggregate.py <root> <arm> [<arm> ...] [--json out.json]
Reads results/<root>/<arm>/pass*/<plan>.json; SLO thresholds match loadgen defaults.
"""

import argparse
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(__file__))
from loadgen import summarize  # noqa: E402

METRICS = ["out_tok_s", "goodput_frac", "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
           "itl_p50", "itl_p99", "worst_stall", "maxgap_p90", "errors"]


def slo():
    return argparse.Namespace(ttft_slo=1.0, ttft_slo_long=3.0, tpot_slo=0.1, gap_slo=0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    base = f"/root/workspace/muxpd-eval/results/{a.root}"
    plans = sorted({os.path.basename(p)[:-5] for arm in a.arms
                    for p in glob.glob(f"{base}/{arm}/pass*/*.json")})
    out = {}
    for plan in plans:
        for arm in a.arms:
            files = sorted(glob.glob(f"{base}/{arm}/pass*/{plan}.json"))
            if not files:
                continue
            runs = [summarize(json.load(open(f)), slo()) for f in files]
            row = {"passes": len(runs)}
            for m in METRICS:
                vals = [r[m] for r in runs if r[m] is not None]
                if vals:
                    row[m] = {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}
            out.setdefault(plan, {})[arm] = row
            f = lambda m, k="mean", d=3: ("%." + str(d) + "f") % row[m][k]  # noqa: E731
            print(f"{plan:14s} {arm:12s} n={row['passes']} err={int(row['errors']['max'])} "
                  f"out={f('out_tok_s', d=1)}[{f('out_tok_s','min',1)}-{f('out_tok_s','max',1)}] "
                  f"good={f('goodput_frac')}[{f('goodput_frac','min')}-{f('goodput_frac','max')}] "
                  f"ttft50={f('ttft_p50')}[{f('ttft_p50','min')}-{f('ttft_p50','max')}] "
                  f"ttft99={f('ttft_p99')} tpot50={f('tpot_p50',d=4)} "
                  f"itl99={f('itl_p99')}[{f('itl_p99','min')}-{f('itl_p99','max')}] "
                  f"stall={f('worst_stall')}[{f('worst_stall','min')}-{f('worst_stall','max')}]")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
