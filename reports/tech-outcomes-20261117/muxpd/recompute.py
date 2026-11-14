"""Recompute wl38 evidence with Python's standard library; never executes archived code.

python recompute.py --write   # write versioned derived metrics
python recompute.py --check   # default: exact semantic comparison with saved metrics
All paths resolve relative to this file, or --root. Source evidence is read-only.
"""

import argparse
import gzip
import hashlib
import json
import math
import sys
from itertools import pairwise
from pathlib import Path

WORKLOADS = (
    "wl-docchat-r1.3",
    "wl-2wiki-r1.6",
    "wl-hotpot-r1.0",
    "wl-govsum-r1.0",
    "wl-code-r1.0",
    "wl-chat-r1.5",
)
SLO = {"ttft_short_s": 1.0, "ttft_long_s": 3.0, "tpot_s": 0.1, "maxgap_s": 0.5}
CAVEAT = (
    "Events timestamp nonempty client-side SSE text receipts. Empty or coalesced "
    "text means receipt counts need not equal completion token counts; receipt gaps "
    "are not engine token timings. Outputs are forced with ignore_eos; no answer "
    "quality is measured. Each arm ran once, MuxPD before native; no significance claim."
)


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(
            stream, parse_constant=lambda s: _invalid(f"nonfinite JSON number: {s}")
        )


def _invalid(message):
    raise ValueError(message)


def finite(value, context):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        _invalid(f"{context}: expected a finite number")
    return value


def percentile(values, p):
    """Linear interpolation at (n-1)*p/100, matching the original load generator."""
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def quantiles(values):
    return {f"p{p}": percentile(values, p) for p in (50, 95, 99)}


def summarize(result, manifest):
    """Validate all planned requests before evaluating the unchanged conjunctive SLO."""
    planned = manifest["requests"]
    expected = {r["id"]: r for r in planned}
    rows = result["rows"]
    if len(expected) != len(planned) or len(rows) != len(planned) or not planned:
        _invalid("missing/duplicate planned or measured requests")
    if len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != set(
        expected
    ):
        _invalid("measured request identities differ from the frozen plan")
    t0 = finite(result["t0"], "t0")
    if "params" in manifest and result.get("params") != manifest["params"]:
        _invalid("measured workload parameters differ from the frozen plan")
    for row in rows:
        exp = expected[row["id"]]
        context = f"request {row['id']}"
        for key in ("kind", "in", "out"):
            if row[key] != exp[key]:
                _invalid(f"{context}: {key} differs from the frozen plan")
        scheduled = finite(row["sched"], f"{context} scheduled time")
        finite(row["sent"], f"{context} send time")
        if abs(scheduled - t0 - exp["t"]) > 1e-7:
            _invalid(f"{context}: scheduled arrival differs from the frozen plan")
        events = row["events"]
        for value in events:
            finite(value, f"{context} receipt time")
        if any(b < a for a, b in pairwise(events)) or (
            events and events[0] < scheduled
        ):
            _invalid(
                f"{context}: receipt times are backwards or precede scheduled arrival"
            )
        usage = row["usage"]
        successful = not row["error"] and bool(events)
        if successful and not usage:
            _invalid(f"{context}: completed request lacks token usage")
        if usage:
            for key, want in (
                ("prompt_tokens", exp["in"]),
                ("completion_tokens", exp["out"]),
                ("total_tokens", exp["in"] + exp["out"]),
            ):
                if usage.get(key) != want:
                    _invalid(f"{context}: usage {key} differs from the frozen plan")
    summary = _metrics(rows, t0)
    summary["by_kind"] = {
        kind: _metrics([r for r in rows if r["kind"] == kind], t0)
        for kind in sorted({r["kind"] for r in rows})
    }
    return summary


def _metrics(rows, t0):
    ok = [r for r in rows if not r["error"] and r["events"]]
    values = {"ttft_s": [], "tpot_s": [], "itl_s": [], "maxgap_s": []}
    violations = {"error": len(rows) - len(ok), "ttft": 0, "tpot": 0, "gap": 0}
    good = 0
    for row in ok:
        events = row["events"]
        gaps = [b - a for a, b in pairwise(events)]
        ttft = events[0] - row["sched"]
        tokens = row["usage"]["completion_tokens"]
        tpot = (events[-1] - events[0]) / (tokens - 1) if tokens > 1 else 0.0
        maxgap = max(gaps, default=0.0)
        fail = {
            "ttft": ttft
            > (SLO["ttft_long_s"] if row["kind"] == "long" else SLO["ttft_short_s"]),
            "tpot": tpot > SLO["tpot_s"],
            "gap": maxgap > SLO["maxgap_s"],
        }
        for key, failed in fail.items():
            violations[key] += int(failed)
        good += not any(fail.values())
        for key, value in (("ttft_s", ttft), ("tpot_s", tpot), ("maxgap_s", maxgap)):
            values[key].append(value)
        values["itl_s"].extend(gaps)
    span = max((r["events"][-1] for r in ok), default=t0) - t0
    out_tokens = sum(r["usage"]["completion_tokens"] for r in ok)
    in_tokens = sum(r["in"] for r in ok)

    def rate(n):
        return n / span if span > 0 else None

    return {
        "requests": len(rows),
        "completed": len(ok),
        "errors": len(rows) - len(ok),
        "good": good,
        "goodput_frac": good / len(rows),
        "span_s": span,
        "completion_tokens": out_tokens,
        "prompt_tokens": in_tokens,
        "out_tok_s": rate(out_tokens),
        "total_tok_s": rate(out_tokens + in_tokens),
        "req_s": rate(len(ok)),
        "goodput_req_s": rate(good),
        **{key: quantiles(val) for key, val in values.items()},
        # Explicit accurate summation is stable across Python 3.11/3.12;
        # builtin sum(float) changed algorithms in Python 3.12.
        "tpot_mean_s": math.fsum(values["tpot_s"]) / len(ok) if ok else None,
        "worst_stall_s": max(values["itl_s"], default=None),
        "send_lag_max_s": max(r["sent"] - r["sched"] for r in rows),
        "violations": violations,
        "sse_events": sum(len(r["events"]) for r in ok),
        "sse_event_count_equals_completion_tokens": sum(
            len(r["events"]) == r["usage"]["completion_tokens"] for r in ok
        ),
    }


def classify(comparisons):
    deltas = [c["delta"] for c in comparisons.values()]
    errors = sum(
        c[a]["errors"] for c in comparisons.values() for a in ("native", "mux43")
    )
    if any(d["goodput_pp"] <= -3 or d["out_tok_pct"] <= -3 for d in deltas):
        return "cost"
    if (
        all(d["goodput_pp"] >= 15 and d["out_tok_pct"] >= -2 for d in deltas)
        and not errors
    ):
        return "very good"
    if all(d["goodput_pp"] >= 5 for d in deltas) or all(
        d["out_tok_pct"] >= 5 for d in deltas
    ):
        return "gain"
    return "flat"


def verify_originals(root):
    """Check archived byte hashes when the packaging manifest is present."""
    path = root / "originals-manifest.json"
    if not path.exists():
        return
    for item in read_json(path)["files"]:
        target = root / item["path"]
        if not target.resolve().is_relative_to(root.resolve()):
            _invalid(f"original manifest path escapes report: {item['path']}")
        stored = target.read_bytes()
        if (
            len(stored) != item["bytes"]
            or hashlib.sha256(stored).hexdigest() != item["sha256"]
        ):
            _invalid(f"original storage checksum mismatch: {item['path']}")
        original = gzip.decompress(stored) if item.get("encoding") == "gzip" else stored
        if (
            len(original) != item["original_bytes"]
            or hashlib.sha256(original).hexdigest() != item["original_sha256"]
        ):
            _invalid(f"original decoded checksum mismatch: {item['path']}")


def recompute(root):
    root = Path(root)
    verify_originals(root)
    workloads = {}
    requests, errors = 0, 0
    for name in WORKLOADS:
        manifest = read_json(root / f"data/request-manifests/{name}.json")
        comparisons = {}
        for tp in (2, 4):
            arms = {}
            for arm in ("native", "mux43"):
                result = read_json(
                    root / f"data/results/wl38-tp{tp}/{arm}/pass1/{name}.json.gz"
                )
                arms[arm] = summarize(result, manifest)
                requests += arms[arm]["requests"]
                errors += arms[arm]["errors"]
            n, m = arms["native"], arms["mux43"]
            if n["out_tok_s"] is None or n["out_tok_s"] <= 0 or m["out_tok_s"] is None:
                _invalid(
                    f"{name} tp{tp}: cannot compare output throughput without completed requests"
                )
            comparisons[f"tp{tp}"] = {
                **arms,
                "delta": {
                    "goodput_pp": (m["good"] - n["good"]) / n["requests"] * 100,
                    "out_tok_pct": (m["out_tok_s"] / n["out_tok_s"] - 1) * 100,
                },
            }
        workloads[name] = {
            "plan_sha256": manifest["plan_sha256"],
            "params": manifest["params"],
            "comparisons": comparisons,
            "classification": classify(comparisons),
        }
    return {
        "schema_version": 1,
        "scope": "wl38; six workloads; one pass per TP2/TP4 native/MuxPD arm",
        "measurement_caveat": CAVEAT,
        "slo": SLO,
        "selection_rule": {
            "cost": "any TP goodput <= -3pp or throughput <= -3%",
            "very good": "both TPs goodput >= +15pp, throughput >= -2%, zero errors",
            "gain": "both TPs goodput >= +5pp or both throughputs >= +5%",
            "flat": "otherwise; cost takes priority",
        },
        "totals": {
            "requests_across_all_arms": requests,
            "errors_across_all_arms": errors,
        },
        "workloads": workloads,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument(
        "--check", action="store_true", help="default: check saved metrics exactly"
    )
    args = parser.parse_args(argv)
    try:
        actual = recompute(args.root)
        target = args.root / "metrics-v1.json"
        if args.write:
            target.write_text(
                json.dumps(actual, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                f"Wrote {target}: 24 runs, {actual['totals']['requests_across_all_arms']} requests"
            )
        elif read_json(target) != actual:
            _invalid(
                f"{target}: saved metrics differ; inspect evidence and use --write to regenerate"
            )
        else:
            print("PASS: all 24 runs, request identities and metrics-v1.json match")
    except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
