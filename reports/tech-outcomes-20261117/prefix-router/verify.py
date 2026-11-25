"""Offline check of published bytes, per-request counters, per-window P95, and reported ratios."""

from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict

root = Path(__file__).resolve().parent
data = root / "swe-c4"


def check(ok, msg):
    if not ok:
        raise ValueError(msg)


def same(x, y):
    check(math.isclose(float(x), float(y), rel_tol=1e-10, abs_tol=1e-8), (x, y))


def rows(name):
    with (data / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def q95(xs):
    return statistics.quantiles(xs, n=100, method="inclusive")[94]


for line in (root / "SHA256SUMS").read_text().splitlines():
    want, rel = line.split("  ", 1)
    check(hashlib.sha256((root / rel).read_bytes()).hexdigest() == want, rel)
for item in json.loads((root / "originals-manifest.json").read_text())["files"]:
    if item["included_verbatim"]:
        b = (data / item["path"]).read_bytes()
        check(
            len(b) == item["bytes"] and hashlib.sha256(b).hexdigest() == item["sha256"],
            item["path"],
        )
windows = rows("windows.csv")
groups = defaultdict(list)
for r in rows("timings.csv"):
    groups[r["window"]].append(r)
keys = {f"r{i}-{p}" for i in (1, 2, 3) for p in ("random", "prefix", "lifecycle")}
check(
    set(groups) == keys
    and len(windows) == 9
    and {f"r{w['rep']}-{w['policy']}" for w in windows} == keys,
    "9-window matrix",
)
by_policy = defaultdict(lambda: defaultdict(list))
for w in windows:
    key = f"r{w['rep']}-{w['policy']}"
    rr = groups[key]
    summary = json.loads((data / key / "client/summary.json").read_text())
    active = [r for r in rr if float(r["end"]) <= 900 and int(r["success"])]
    same(len(rr), 34)
    same(len(active), 30)
    same(sum(int(r["success"]) for r in rr), 34)
    same(summary["measurement_seconds"], 900)
    same(summary["failed_requests"], 0)
    tokens = sum(int(r["in_window"]) for r in rr)
    same(tokens, summary["observed_output_tokens_in_window"])
    same(tokens, w["window_tokens"])
    vals = dict(
        output_tok_s=tokens / 900,
        requests_s=len(active) / 900,
        ttft_p95_ms=q95(
            [(float(r["first"]) - float(r["start"])) * 1000 for r in active]
        ),
        tpot_p95_ms=q95(
            [
                (float(r["last"]) - float(r["first"])) * 1000 / (int(r["output"]) - 1)
                for r in active
                if int(r["output"]) > 1
            ]
        ),
        e2e_p95_ms=q95([(float(r["last"]) - float(r["start"])) * 1000 for r in active]),
    )
    same(vals["ttft_p95_ms"], summary["ttft_seconds_p95"] * 1000)
    for k, v in vals.items():
        same(v, w[k])
        by_policy[w["policy"]][k].append(v)
for c in rows("comparisons.csv"):
    metric = c["metric"]
    same(
        100
        * (
            statistics.mean(by_policy[c["numerator"]][metric])
            / statistics.mean(by_policy[c["denominator"]][metric])
            - 1
        ),
        c["percent_change"],
    )
for p, m in by_policy.items():
    print(
        p,
        "tok/s",
        round(statistics.mean(m["output_tok_s"]), 3),
        "TTFT P95 ms",
        round(statistics.mean(m["ttft_p95_ms"]), 2),
    )
print("OK: 9 windows; 270 in-window completions + 36 drained; 306 successful requests.")
