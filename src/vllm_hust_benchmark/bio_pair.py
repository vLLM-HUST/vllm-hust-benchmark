"""Counterbalanced within-engine comparisons of a single boolean task option."""

from collections import defaultdict
import json
from statistics import mean, median

import numpy as np


def validate_pairs(spec, cases):
    by_id = {case["id"]: case for case in cases}
    names = [item for pair in spec["pairs"].values() for item in pair]
    if len(set(names)) != len(names) or set(names) != set(by_id):
        raise ValueError("Pair membership must partition the complete suite")
    if len({case["profile_id"] for case in cases}) != 1:
        raise ValueError("Paired starts must share one runtime profile")
    for baseline, candidate in spec["pairs"].values():
        normalized = []
        for case_id, arm in [(baseline, False), (candidate, True)]:
            case = by_id[case_id].copy()
            case.pop("id")
            opts = case["options"].copy()
            if opts.pop(spec["option"], None) is not arm:
                raise ValueError("Pair option or arm identity differs")
            normalized.append(case | {"options": opts})
        if normalized[0] != normalized[1]:
            raise ValueError("Pair inputs, task or additional options differ")


def order_cases(spec, cases, start):
    validate_pairs(spec, cases)
    by_id = {case["id"]: case for case in cases}
    ordered, membership = [], {}
    for name, pair in spec["pairs"].items():
        for key in pair[:: -1 if start % 2 else 1]:
            membership[key] = name
            ordered.append(by_id[key])
    return ordered, membership


def score_difference(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("Paired score batches differ")
    largest = 0.0
    for a, b in zip(left, right, strict=True):
        metrics = {"token_logprobs", "sum_logprob", "mean_logprob", "mean_nll"}
        if {k: v for k, v in a.items() if k not in metrics} != {
            k: v for k, v in b.items() if k not in metrics
        }:
            raise ValueError("Paired score targets or masks differ")
        x, y = (np.asarray(row["token_logprobs"], dtype=float) for row in (a, b))
        if x.shape != y.shape or x.size == 0 or not np.isfinite(x - y).all():
            raise ValueError("Invalid paired score arrays")
        largest = max(largest, float(np.abs(x - y).max()))
    return largest


def paired_results(root, manifest, suite, cases):
    """Numerical reference gates and all request clocks are checked by summarize."""
    spec = suite["pairing"]
    by_pair = defaultdict(list)
    for job in manifest["runs"]:
        if job["fault"]:
            continue
        start = job["start_index"]
        planned, membership = order_cases(spec, list(cases.values()), start)
        d = root / job["directory"]
        native = json.loads((d / "native.json").read_text())
        rows = [json.loads(s) for s in (d / "requests.jsonl").read_text().splitlines()]
        if [r["case_id"] for r in rows if r["phase"] == "first"] != [
            c["id"] for c in planned
        ]:
            raise ValueError("Actual pair order differs from the counterbalanced plan")
        if any(r.get("pair_id") != membership[r["case_id"]] for r in rows):
            raise ValueError("Raw request pair ID differs")
        for pair, (a, b) in spec["pairs"].items():
            inputs = [native["inputs"][key] for key in (a, b)]
            if {k: v for k, v in inputs[0].items() if k != "sample_ids"} != {
                k: v for k, v in inputs[1].items() if k != "sample_ids"
            }:
                raise ValueError("Prepared pair inputs differ")
            sides = [
                [r for r in rows if r["case_id"] == key and r["phase"] == "steady"]
                for key in (a, b)
            ]
            point = {"start": start, "order": "BA" if start % 2 else "AB"}
            errors = []
            for left, right in zip(*sides, strict=True):
                if (
                    left["work"] != right["work"]
                    or left["repetition"] != right["repetition"]
                ):
                    raise ValueError("Paired workload or repetition differs")
                if spec.get("output_check") == "conditional_logprobs_v1":
                    errors.append(score_difference(left["output"], right["output"]))
            if errors:
                point["token_max_abs"] = max(errors)
                if point["token_max_abs"] > suite["gates"]["selected_logprob_max"]:
                    raise ValueError("Cross-arm score difference exceeds protocol")
            for arm, rr in zip(("baseline", "candidate"), sides, strict=True):
                ms = [(r["finished_ns"] - r["started_ns"]) / 1e6 for r in rr]
                seconds = (rr[-1]["finished_ns"] - rr[0]["started_ns"]) / 1e9
                point[arm] = dict(
                    p50_ms=median(ms),
                    p90_ms=float(np.quantile(ms, 0.9)),
                    units_per_second=sum(r["work"][spec["work_counter"]] for r in rr)
                    / seconds,
                    peak_allocated_bytes=max(
                        r["memory_after"]["peak_allocated_bytes"] for r in rr
                    ),
                    peak_reserved_bytes=max(
                        r["memory_after"]["peak_reserved_bytes"] for r in rr
                    ),
                )
            point["speedup"] = (
                point["baseline"]["p50_ms"] / point["candidate"]["p50_ms"]
            )
            by_pair[pair].append(point)
    result = {}
    for pair, points in by_pair.items():
        ratios = [p["speedup"] for p in points]
        # Resample engine starts as pairs. Never count repeated calls as independent starts.
        rng = np.random.default_rng(1729)
        draws = rng.choice(ratios, size=(10000, len(ratios)), replace=True).mean(axis=1)
        result[pair] = dict(
            starts=points,
            mean_paired_speedup=mean(ratios),
            bootstrap95=np.quantile(draws, [0.025, 0.975]).tolist(),
            work_unit=spec["work_counter"],
        )
    return result


def paired_markdown(results):
    lines = [
        "",
        "Paired steady measurements (baseline / candidate latency ratio; 95% paired-start bootstrap intervals).",
        "",
        "| Pair | Start / order | p50 ms (A / B) | p90 ms (A / B) | Units/s (A / B) | Peak allocated MiB (A / B) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in results.items():
        for p in r["starts"]:
            a, b = p["baseline"], p["candidate"]
            lines.append(
                f"| {name} | {p['start']} / {p['order']} | {a['p50_ms']:.3f} / {b['p50_ms']:.3f} | {a['p90_ms']:.3f} / {b['p90_ms']:.3f} | {a['units_per_second']:.2f} / {b['units_per_second']:.2f} | {a['peak_allocated_bytes'] / 2**20:.1f} / {b['peak_allocated_bytes'] / 2**20:.1f} |"
            )
    lines.append("")
    for name, r in results.items():
        lo, hi = r["bootstrap95"]
        lines.append(f"- {name}: {r['mean_paired_speedup']:.3f}x [{lo:.3f}, {hi:.3f}]")
    return lines
