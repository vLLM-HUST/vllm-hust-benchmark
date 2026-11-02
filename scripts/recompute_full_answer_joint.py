"""Recompute public full-answer metrics without publishing private task text.

This checks scalar projections, not the original answers or device telemetry.
The separately retained raw-response audit binds those records by SHA256.
"""

import argparse
import json
import math
from pathlib import Path
import statistics


def positive(value):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("Expected a finite positive measurement")
    return value


def percentile(values, probability):
    return sorted(values)[math.ceil(len(values) * probability) - 1]


def arm_metrics(arm):
    rows = arm["rows"]
    if len(rows) != 200 or len({r["case_id"] for r in rows}) != 200:
        raise ValueError("Expected the complete fixed200 set")
    for row in rows:
        if type(row["correct"]) is not bool or type(row["parsed"]) is not bool:
            raise ValueError("Quality indicators must be booleans")
        if row["correct"] and not row["parsed"]:
            raise ValueError("An unparsed answer cannot be correct")
        if row["finish_reason"] not in ("stop", "length"):
            raise ValueError("Incomplete response")
        if not 0 < row["output_tokens"] <= 8192:
            raise ValueError("Output budget mismatch")
        for key in ("ttft_ms", "e2e_ms"):
            positive(row[key])
        if row["ttft_ms"] > row["e2e_ms"]:
            raise ValueError("Invalid request clock ordering")
        expected_tpot = (
            (row["e2e_ms"] - row["ttft_ms"]) / (row["output_tokens"] - 1)
            if row["output_tokens"] > 1
            else None
        )
        if expected_tpot is None:
            if row["tpot_ms"] is not None:
                raise ValueError("Single-token response has no TPOT")
        elif not math.isclose(row["tpot_ms"], expected_tpot, rel_tol=1e-8):
            raise ValueError("TPOT does not match the response interval")
    seconds = positive(arm["phase_seconds"])
    price = arm["cost"]["price_basis"]
    if price["basis"] != "public_list_price_estimate_not_actual_invoice":
        raise ValueError("Price scope changed")
    if price["accelerator_count"] != 2:
        raise ValueError("Expected the same reserved two-card resource envelope")
    rate = (
        positive(price["card_monthly_equivalent"])
        * 2
        / (positive(price["month_days_for_analysis"]) * 24 * 3600)
    )
    tokens = sum(r["output_tokens"] for r in rows)
    timing = {}
    for key in ("ttft_ms", "tpot_ms", "e2e_ms"):
        values = [r[key] for r in rows if r[key] is not None]
        timing[key] = dict(
            count=len(values),
            mean=statistics.fmean(values),
            p95=percentile(values, 0.95),
            p99=percentile(values, 0.99),
        )
    return dict(
        completed=200,
        failed=0,
        correct=sum(r["correct"] for r in rows),
        truncated=sum(r["finish_reason"] == "length" for r in rows),
        quality_by_finish={
            finish: dict(
                total=sum(r["finish_reason"] == finish for r in rows),
                correct=sum(
                    r["finish_reason"] == finish and r["correct"] for r in rows
                ),
            )
            for finish in ("stop", "length")
        },
        successful_output_tokens=tokens,
        output_tokens_per_second=tokens / seconds,
        timing=timing,
        request_phase_cny=rate * seconds,
        cny_per_million_output_tokens=rate * seconds / tokens * 1e6,
    )


def paired_metrics(data):
    left, right = data["arms"]
    if (left["role"], right["role"]) != ("B0", "B1"):
        raise ValueError("Expected ordered B0/B1")
    if (
        left["runtime"]["vllm_version"] != "0.18.0"
        or left["runtime"]["ascend_version"] != "0.18.0"
    ):
        raise ValueError("B0 is fixed at 0.18/0.18")
    for key in ("original_quality_plan_sha256", "token_preflight_sha256"):
        if left[key] != right[key]:
            raise ValueError("Input identity mismatch")
    if left["cost"]["price_basis"] != right["cost"]["price_basis"]:
        raise ValueError("Cost comparison needs the same price basis")
    pairs = list(zip(left["rows"], right["rows"], strict=True))
    if any(a["case_id"] != b["case_id"] for a, b in pairs):
        raise ValueError("Paired task order mismatch")
    a, b = arm_metrics(left), arm_metrics(right)
    return dict(
        arms=[a, b],
        paired_quality=dict(
            improved=sum(not x["correct"] and y["correct"] for x, y in pairs),
            regressed=sum(x["correct"] and not y["correct"] for x, y in pairs),
            difference_percentage_points=(b["correct"] - a["correct"]) / 2,
        ),
        output_throughput_change_percent=(
            b["output_tokens_per_second"] / a["output_tokens_per_second"] - 1
        )
        * 100,
        rental_cost_reduction_percent=(
            1 - b["cny_per_million_output_tokens"] / a["cny_per_million_output_tokens"]
        )
        * 100,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projection", type=Path)
    args = parser.parse_args()
    print(json.dumps(paired_metrics(json.loads(args.projection.read_text())), indent=2))


if __name__ == "__main__":
    main()
