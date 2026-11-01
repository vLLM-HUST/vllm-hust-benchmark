"""Recompute B0-relative metrics from an immutable, existing paired record."""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "reports/full-answer-joint-20261030/joint-summary.json"
OUTPUT = "reports/b0-baseline-metrics-20261101/summary.json"


def relative_metrics(o0, t0, o1, t1, e0, e1):
    values = (o0, t0, o1, t1, e0, e1)
    if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError(
            "Tokens, seconds and allocated expenses must be finite and positive"
        )
    q0, q1 = o0 / t0, o1 / t1
    speed = q1 / q0
    return {
        "throughput_ratio": speed,
        "eta_b0_percent": 100 * speed,
        "throughput_gain_percent": 100 * (speed - 1),
        "batch_time_ratio": t0 / t1,
        "batch_time_reduction_percent": 100 * (1 - t1 / t0),
        "equal_output_tokens": o0 == o1,
        "cost_reduction_percent": 100 * (1 - (e1 / o1) / (e0 / o0)),
    }


def recompute(root=ROOT):
    raw = (root / SOURCE).read_bytes()
    source = json.loads(raw)
    a, b = source["arms"]
    if [a["role"], b["role"]] != ["B0", "B1"]:
        raise ValueError("The frozen source must contain B0 followed by B1")
    inputs = []
    for arm in (a, b):
        tokens, seconds = arm["successful_output_tokens"], arm["phase_seconds"]
        if not math.isclose(
            tokens / seconds, arm["output_tokens_per_second"], rel_tol=1e-12
        ):
            raise ValueError(
                "Reported throughput disagrees with tokens and elapsed time"
            )
        if sum(r["output_tokens"] for r in arm["rows"]) != tokens:
            raise ValueError("Request-level token sum disagrees with the aggregate")
        inputs.append(
            {
                key: arm[key]
                for key in (
                    "role",
                    "run",
                    "completed",
                    "failed",
                    "correct",
                    "successful_output_tokens",
                    "phase_seconds",
                    "output_tokens_per_second",
                )
            }
            | {"allocated_expense_cny": arm["cost"]["request_phase_cny"]}
        )
    result = relative_metrics(
        a["successful_output_tokens"],
        a["phase_seconds"],
        b["successful_output_tokens"],
        b["phase_seconds"],
        a["cost"]["request_phase_cny"],
        b["cost"]["request_phase_cny"],
    )
    for key, original in (
        ("throughput_gain_percent", "output_throughput_change_percent"),
        ("cost_reduction_percent", "rental_cost_reduction_percent"),
    ):
        if not math.isclose(result[key], source[original], rel_tol=1e-12):
            raise ValueError("Recomputation disagrees with the original paired report")
    result["latency_reduction_percent"] = {
        key: 100 * (1 - b["timing"][key]["mean"] / a["timing"][key]["mean"])
        for key in ("ttft_ms", "tpot_ms")
    }
    return {
        "schema": "b0-relative-performance/v1",
        "method": "docs/B0_BASELINE_METRICS.md",
        "source": {"path": SOURCE, "sha256": hashlib.sha256(raw).hexdigest()},
        "baseline_eta_b0_percent": 100,
        "protocol": source["protocol"],
        "arms": inputs,
        "metrics": result,
        "interpretation": "Measured B0-relative throughput; separate from calibrated eta_ref",
        "cost_basis": "Same-rate rental estimate over the measured request interval",
        "independent_pair_count": 1,
        "new_model_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Compare with the committed result"
    )
    args = parser.parse_args()
    result = recompute()
    if args.check:
        if result != json.loads((ROOT / OUTPUT).read_text()):
            raise SystemExit(
                "Committed B0-relative result differs from source recomputation"
            )
        print("PASS: source identity, request token sums and B0-relative formulas")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
