#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P2-3 汇总：NIAH 准确率 + 长生成退化，带 Wilson CI 与 bootstrap 差值 CI。

产物：out/stability.json、out/stability.csv
判据（来自 README 预注册）：
  C1 单针 |Δ| ≥ 5 pp 且两臂 Wilson 95% CI 不重叠
  C2 多针同 C1
  C3 复现率差 ≥ 10 pp，或 distinct-4 相对下降 ≥ 20%
"""

import json
import math
import os
import random
import sys


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    out = []
    if not os.path.isfile(path):
        return out
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def boot_diff_ci(a, b, seed=20261105, iters=10000):
    """a、b 是 0/1 列表；返回 Δ=a−b 的 95% bootstrap CI。"""
    if not a or not b:
        return (None, None, None)
    rng = random.Random(seed)
    obs = sum(a) / len(a) - sum(b) / len(b)
    ds = []
    for _ in range(iters):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        ds.append(sum(ra) / len(ra) - sum(rb) / len(rb))
    ds.sort()
    return obs, ds[int(0.025 * iters)], ds[int(0.975 * iters) - 1]


def summarize_niah(recs):
    out = {}
    for kind in ("single", "multi"):
        rs = [r for r in recs if r["kind"] == kind and not r.get("error")]
        if not rs:
            continue
        for L in sorted({r["length"] for r in rs}):
            cell = [r for r in rs if r["length"] == L]
            k = sum(1 for r in cell if r["hit"])
            lo, hi = wilson(k, len(cell))
            out.setdefault(kind, {})[str(L)] = {
                "n": len(cell), "hit": k, "acc": k / len(cell),
                "wilson_lo": lo, "wilson_hi": hi,
                "invalid": sum(1 for r in cell if r.get("invalid")),
                "truncated": sum(1 for r in cell if r.get("truncated")),
                "errors": sum(1 for r in recs if r["kind"] == kind and r["length"] == L and r.get("error")),
            }
            if kind == "single":
                by_depth = {}
                for d in sorted({r["depth"] for r in cell if r["depth"] is not None}):
                    sub = [r for r in cell if r["depth"] == d]
                    by_depth["%.2f" % d] = {
                        "n": len(sub), "hit": sum(1 for r in sub if r["hit"]),
                        "acc": sum(1 for r in sub if r["hit"]) / len(sub),
                    }
                out[kind][str(L)]["by_depth"] = by_depth
    return out


def summarize_longgen(recs):
    out = {}
    for L in sorted({r["length"] for r in recs if not r.get("error")}):
        cell = [r for r in recs if r["length"] == L and not r.get("error")]
        if not cell:
            continue
        def mean(key):
            xs = [r[key] for r in cell if r.get(key) is not None]
            return sum(xs) / len(xs) if xs else None
        lens = sorted(r["out_chars"] for r in cell)
        out[str(L)] = {
            "n": len(cell),
            "exact_rate": sum(1 for r in cell if r["exact_substr"]) / len(cell),
            "rouge_l": mean("rouge_l"),
            "distinct4": mean("distinct4"),
            "repeat10": mean("repeat10"),
            "out_chars_median": lens[len(lens) // 2] if lens else None,
            "invalid": sum(1 for r in cell if r.get("invalid")),
            "truncated": sum(1 for r in cell if r.get("truncated")),
            "errors": sum(1 for r in recs if r["length"] == L and r.get("error")),
        }
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "out")
    arms = ("w8a8", "w4a8")
    niah = {a: load(os.path.join(out_dir, "niah_%s.jsonl" % a)) for a in arms}
    lgen = {a: load(os.path.join(out_dir, "longgen_%s.jsonl" % a)) for a in arms}

    niah_sum = {a: summarize_niah(niah[a]) for a in arms}
    lgen_sum = {a: summarize_longgen(lgen[a]) for a in arms}

    result = {"niah": niah_sum, "longgen": lgen_sum, "criteria": {}}

    print("\n=== NIAH (hit / n, Wilson 95% CI) ===")
    for kind in ("single", "multi"):
        for L in sorted({k for a in arms for k in niah_sum[a].get(kind, {})}, key=int):
            row = []
            for a in arms:
                c = niah_sum[a].get(kind, {}).get(L)
                row.append("%s %d/%d=%.1f%% [%.1f,%.1f]" % (
                    a, c["hit"], c["n"], 100 * c["acc"],
                    100 * c["wilson_lo"], 100 * c["wilson_hi"]) if c else "%s n/a" % a)
            print("%-7s L=%-6s %s" % (kind, L, " | ".join(row)))

    # ---- 预注册判据 ----
    for kind, key in (("single", "C1"), ("multi", "C2")):
        detail = {}
        triggered = False
        lengths = sorted({k for a in arms for k in niah_sum[a].get(kind, {})}, key=int)
        for L in lengths:
            ca = [1 if r["hit"] else 0 for r in niah["w8a8"]
                  if r["kind"] == kind and r["length"] == int(L) and not r.get("error")]
            cb = [1 if r["hit"] else 0 for r in niah["w4a8"]
                  if r["kind"] == kind and r["length"] == int(L) and not r.get("error")]
            d, lo, hi = boot_diff_ci(cb, ca)
            sa, sb = niah_sum["w8a8"][kind][L], niah_sum["w4a8"][kind][L]
            overlap = not (sb["wilson_lo"] > sa["wilson_hi"] or sa["wilson_lo"] > sb["wilson_hi"])
            hit = (d is not None and abs(d) >= 0.05 and not overlap)
            triggered = triggered or hit
            detail[L] = {"delta_w4a8_minus_w8a8": d, "boot_lo": lo, "boot_hi": hi,
                         "wilson_overlap": overlap, "triggered": hit}
        result["criteria"][key] = {"triggered": triggered, "by_length": detail}

    print("\n=== long generation ===")
    for L in sorted({k for a in arms for k in lgen_sum[a]}, key=int):
        for a in arms:
            c = lgen_sum[a].get(L)
            if c:
                print("L=%-6s %-5s exact=%.1f%% rouge=%.3f distinct4=%.3f repeat10=%.3f chars_med=%d invalid=%d trunc=%d" % (
                    L, a, 100 * c["exact_rate"], c["rouge_l"] or 0, c["distinct4"] or 0,
                    c["repeat10"] or 0, c["out_chars_median"] or 0, c["invalid"], c["truncated"]))

    c3 = {}
    for L in sorted(lgen_sum["w8a8"], key=int):
        a, b = lgen_sum["w8a8"].get(L), lgen_sum["w4a8"].get(L)
        if not a or not b:
            continue
        d_exact = b["exact_rate"] - a["exact_rate"]
        d_d4 = (b["distinct4"] - a["distinct4"]) / a["distinct4"] if a["distinct4"] else None
        c3[L] = {"delta_exact": d_exact, "rel_delta_distinct4": d_d4,
                 "triggered": abs(d_exact) >= 0.10 or (d_d4 is not None and d_d4 <= -0.20)}
    result["criteria"]["C3"] = {"triggered": any(v["triggered"] for v in c3.values()),
                                "by_length": c3}

    print("\n=== pre-registered criteria ===")
    for k in ("C1", "C2", "C3"):
        print("%s triggered=%s" % (k, result["criteria"][k]["triggered"]))

    with open(os.path.join(out_dir, "stability.json"), "w") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False)
    print("\nwrote", os.path.join(out_dir, "stability.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
