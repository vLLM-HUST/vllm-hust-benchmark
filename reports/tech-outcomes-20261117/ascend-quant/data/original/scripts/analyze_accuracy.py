#!/usr/bin/env python3
"""Paired analysis of MMLU-Pro accuracy runs against the published BF16 baseline.

Reports, per arm:
  * overall accuracy with Wilson 95% CI
  * finished-normally vs length-limited split
  * invalid answers / transport failures
  * 14-category accuracy with Wilson 95% CI and delta vs BF16
  * paired McNemar against BF16 (and W4A8 against W8A8)

Usage:
  python -I analyze_accuracy.py \
    --bf16-csv /data/jxd/whb/reports/mmlu-pro-qwen35-b0-precheck-20261012/results/per-case-outcomes.csv.gz \
    --run w8a8=results/w8a8-run1/per-case-outcomes.csv \
    --run w4a8=results/w4a8-run1/per-case-outcomes.csv \
    --outdir analysis
"""
import argparse
import csv
import gzip
import json
import math
import pathlib
import sys

Z = 1.959963984540054


def read_csv(path):
    path = pathlib.Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        out[r["case_id"]] = {
            "category": r["category"],
            "gold": r["gold"],
            "prediction": r["prediction"],
            "valid": bool(int(r["valid"])),
            "correct": bool(int(r["correct"])),
            "transport_success": bool(int(r["transport_success"])),
            "finish_reason": r["finish_reason"],
        }
    return out


def wilson(k, n):
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def acc_block(rows):
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    return {
        "n": n,
        "correct": k,
        "accuracy": (k / n) if n else None,
        "wilson95": [lo, hi],
        "invalid_answers": sum(not r["valid"] for r in rows),
        "transport_failures": sum(not r["transport_success"] for r in rows),
        "length_limited": sum(r["finish_reason"] == "length" for r in rows),
    }


def split_block(rows):
    fin = [r for r in rows if r["finish_reason"] != "length"]
    lim = [r for r in rows if r["finish_reason"] == "length"]
    out = {}
    for name, part in (("finished", fin), ("length_limited", lim)):
        k = sum(r["correct"] for r in part)
        lo, hi = wilson(k, len(part))
        out[name] = {
            "n": len(part),
            "correct": k,
            "accuracy": (k / len(part)) if part else None,
            "wilson95": [lo, hi],
        }
    return out


def binom_two_sided(m, n):
    """Exact two-sided binomial test, p0=0.5, m successes out of n (log-space)."""
    if n == 0:
        return 1.0
    def logpmf(k):
        return (
            math.lgamma(n + 1)
            - math.lgamma(k + 1)
            - math.lgamma(n - k + 1)
            - n * math.log(2.0)
        )
    k = min(m, n - m)
    # sum_{i=0..k} exp(logpmf(i)), stable by factoring out the max term
    logs = [logpmf(i) for i in range(k + 1)]
    top = max(logs)
    tail = math.exp(top) * sum(math.exp(x - top) for x in logs)
    return min(1.0, 2.0 * tail)


def mcnemar(a, b, name_a, name_b):
    """Paired comparison on the shared case set; rows are {case_id: outcome}."""
    ids = sorted(set(a) & set(b))
    if len(ids) != len(a) or len(ids) != len(b):
        raise SystemExit("case sets differ: %d vs %d, shared %d" % (len(a), len(b), len(ids)))
    both = sum(a[i]["correct"] and b[i]["correct"] for i in ids)
    only_a = sum(a[i]["correct"] and not b[i]["correct"] for i in ids)
    only_b = sum((not a[i]["correct"]) and b[i]["correct"] for i in ids)
    neither = len(ids) - both - only_a - only_b
    n = len(ids)
    delta = (only_a - only_b) / n
    discordant = only_a + only_b
    se = (
        math.sqrt(max(0.0, discordant / n - delta * delta) / n) if n else 0.0
    )
    return {
        "case_set_sha": None,
        "n": n,
        "both_correct": both,
        "only_%s_correct" % name_a: only_a,
        "only_%s_correct" % name_b: only_b,
        "neither_correct": neither,
        "delta_accuracy_%s_minus_%s" % (name_a, name_b): delta,
        "wald95_delta": [delta - Z * se, delta + Z * se],
        "mcnemar_exact_p": binom_two_sided(only_a, discordant),
    }


def fmt_pct(x):
    return "—" if x is None else "%.2f%%" % (100 * x)


def table_md(res):
    lines = []
    lines.append("| arm | n | correct | accuracy | 95% Wilson | finished | length-limited | invalid | transport fail |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|")
    for arm in res["arms"]:
        o = arm["overall"]
        lines.append(
            "| %s | %d | %d | %s | [%s, %s] | %s (n=%d) | %s (n=%d) | %d | %d |"
            % (
                arm["label"],
                o["n"],
                o["correct"],
                fmt_pct(o["accuracy"]),
                fmt_pct(o["wilson95"][0]),
                fmt_pct(o["wilson95"][1]),
                fmt_pct(arm["split"]["finished"]["accuracy"]),
                arm["split"]["finished"]["n"],
                fmt_pct(arm["split"]["length_limited"]["accuracy"]),
                arm["split"]["length_limited"]["n"],
                o["invalid_answers"],
                o["transport_failures"],
            )
        )
    return "\n".join(lines)


def category_md(res, arms):
    cats = sorted(res["arms"][0]["categories"])
    head = "| category | " + " | ".join(
        "%s acc (Δ vs BF16)" % a["label"] for a in res["arms"]
    ) + " |"
    sep = "|---|" + "---:|" * len(res["arms"])
    lines = [head, sep]
    bf = {a["label"]: a for a in res["arms"]}
    for c in cats:
        cells = []
        for a in res["arms"]:
            blk = a["categories"][c]
            delta = ""
            if a["label"] != "bf16":
                d = blk["accuracy"] - bf["bf16"]["categories"][c]["accuracy"]
                delta = " (%+.2f pp)" % (100 * d)
            cells.append("%s (n=%d)%s" % (fmt_pct(blk["accuracy"]), blk["n"], delta))
        lines.append("| %s | %s |" % (c, " | ".join(cells)))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-csv", required=True)
    ap.add_argument("--run", action="append", default=[],
                    help="label=per-case-outcomes.csv (repeatable)")
    ap.add_argument("--outdir", required=True, type=pathlib.Path)
    a = ap.parse_args()

    datasets = {"bf16": read_csv(a.bf16_csv)}
    for spec in a.run:
        label, path = spec.split("=", 1)
        datasets[label] = read_csv(path)

    ref = datasets["bf16"]
    res = {"arms": [], "paired": {}}
    for label, rows in datasets.items():
        block = acc_block(list(rows.values()))
        block.update(split_block(list(rows.values())))
        cats = {}
        for c in sorted({r["category"] for r in rows.values()}):
            part = [r for r in rows.values() if r["category"] == c]
            k = sum(r["correct"] for r in part)
            lo, hi = wilson(k, len(part))
            cats[c] = {
                "n": len(part),
                "correct": k,
                "accuracy": k / len(part),
                "wilson95": [lo, hi],
            }
        res["arms"].append(
            {
                "label": label,
                "overall": block,
                "split": split_block(list(rows.values())),
                "categories": cats,
            }
        )

    for label, rows in datasets.items():
        if label == "bf16":
            continue
        res["paired"]["%s_vs_bf16" % label] = mcnemar(rows, ref, label, "bf16")
    if "w4a8" in datasets and "w8a8" in datasets:
        res["paired"]["w4a8_vs_w8a8"] = mcnemar(
            datasets["w4a8"], datasets["w8a8"], "w4a8", "w8a8"
        )

    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.outdir / "accuracy-analysis.json").write_text(
        json.dumps(res, indent=1, ensure_ascii=False)
    )
    md = ["# MMLU-Pro 12,032 题准确率分析", "", table_md(res), ""]
    md.append("## 分类别")
    md.append("")
    md.append(category_md(res, res["arms"]))
    md.append("")
    md.append("## 配对检验（McNemar，exact）")
    md.append("")
    for key, m in res["paired"].items():
        md.append(
            "- **%s**: n=%d，仅前者对 %d，仅后者对 %d，Δ=%+.3f pp（Wald95 %+.3f..%+.3f pp），exact p=%.3g"
            % (
                key,
                m["n"],
                m["only_%s_correct" % key.split("_vs_")[0]],
                m["only_%s_correct" % key.split("_vs_")[1]],
                100 * m["delta_accuracy_%s_minus_%s" % tuple(key.split("_vs_"))],
                100 * m["wald95_delta"][0],
                100 * m["wald95_delta"][1],
                m["mcnemar_exact_p"],
            )
        )
    (a.outdir / "accuracy-analysis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
