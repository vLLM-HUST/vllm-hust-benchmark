"""Recompute paired burst results from per-request records and device samples."""

import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


def percentile(values, p):
    values = sorted(values)
    rank = (len(values) - 1) * p
    a, b = math.floor(rank), math.ceil(rank)
    return values[a] + (values[b] - values[a]) * (rank - a)


def summarize(values):
    return dict(
        n=len(values),
        mean=statistics.mean(values),
        p95=sorted(values)[math.ceil(0.95 * len(values)) - 1],
        p99=sorted(values)[math.ceil(0.99 * len(values)) - 1],
        p95_rank=math.ceil(0.95 * len(values)),
        p99_rank=math.ceil(0.99 * len(values)),
        p95_linear=percentile(values, 0.95),
        p99_linear=percentile(values, 0.99),
    )


def verify(root, name):
    out = root / name
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert receipt["bench_exit_code"] == 0
    assert receipt["child_exit_codes"] == [0, 0]
    assert receipt["allocated_cards"] == 2 and receipt["reserved_cards"] == 4
    assert receipt["simultaneous_other_services"] == 0
    for path, expected in receipt["input_hashes"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
    r = json.loads((out / "mmlu-pro.json").read_text())
    assert r["completed"] == 200 and r["failed"] == 0
    for field in [
        "input_lens",
        "output_lens",
        "ttfts",
        "latencies",
        "start_times",
        "errors",
    ]:
        assert len(r[field]) == 200, field
    assert not any(r["errors"])
    assert sum(r["output_lens"]) == r["total_output_tokens"]
    assert sum(r["input_lens"]) == r["total_input_tokens"]
    assert all(n > 0 for n in r["output_lens"])
    throughput = sum(r["output_lens"]) / r["duration"]
    assert math.isclose(throughput, r["output_throughput"], rel_tol=1e-10)
    tpot = [
        (latency - ttft) / (n - 1)
        for latency, ttft, n in zip(r["latencies"], r["ttfts"], r["output_lens"])
        if n > 1
    ]
    assert math.isclose(statistics.mean(tpot) * 1000, r["mean_tpot_ms"], rel_tol=1e-8)
    assert math.isclose(
        statistics.mean(r["ttfts"]) * 1000, r["mean_ttft_ms"], rel_tol=1e-8
    )
    a = min(r["start_times"])
    b = max(t + latency for t, latency in zip(r["start_times"], r["latencies"]))
    samples = [
        json.loads(line)
        for line in (out / "device-samples.jsonl").read_text().splitlines()
    ]
    devices = {str(i): dict(weighted_sum=0.0, covered_seconds=0.0) for i in range(4, 8)}
    for sample, following in zip(samples, samples[1:]):
        lo, hi = (
            max(a, sample["end_monotonic"]),
            min(b, following["end_monotonic"], sample["end_monotonic"] + 5),
        )
        if hi <= lo:
            continue
        lines = sample["raw"].splitlines()
        for i, line in enumerate(lines[:-1]):
            match = re.match(r"\|\s*(\d+)\s+910B2\s*\|", line)
            if match and match[1] in devices:
                value = float(lines[i + 1].split("|")[3].split()[0])
                assert 0 <= value <= 100
                d = devices[match[1]]
                d["weighted_sum"] += value * (hi - lo)
                d["covered_seconds"] += hi - lo
    for d in devices.values():
        d["mean_ai_core_percent"] = (
            d.pop("weighted_sum") / d["covered_seconds"]
            if d["covered_seconds"]
            else None
        )
        d["uncovered_seconds"] = b - a - d["covered_seconds"]
    return dict(
        arm=receipt["arm"],
        requests=200,
        failures=0,
        benchmark_duration_s=r["duration"],
        output_tokens=r["total_output_tokens"],
        input_tokens=r["total_input_tokens"],
        output_tps=throughput,
        ttft_s=summarize(r["ttfts"]),
        tpot_s=summarize(tpot),
        timing="Original vLLM bench SSE-chunk timing; reported P95/P99 use nearest rank; original linear estimates retained separately",
        allocated_card_seconds_per_output_token=2
        * r["duration"]
        / r["total_output_tokens"],
        reserved_card_seconds_per_output_token=4
        * r["duration"]
        / r["total_output_tokens"],
        lifecycle_allocated_card_seconds_per_output_token=2
        * (receipt["ended"] - receipt["started"])
        / r["total_output_tokens"],
        request_span_s=b - a,
        device_utilization=devices,
        speculative_decode={k: v for k, v in r.items() if k.startswith("spec_decode_")},
    )


def compare(root, b0_name, b1_name):
    b0, b1 = verify(root, b0_name), verify(root, b1_name)
    a = json.loads((root / b0_name / "mmlu-pro.json").read_text())
    b = json.loads((root / b1_name / "mmlu-pro.json").read_text())
    assert a["input_lens"] == b["input_lens"], "Actual input lengths differ"
    return dict(
        project="vLLM-HUST",
        dataset="MMLU-Pro",
        b0=b0,
        b1=b1,
        configuration="Original dataset B0 and school E16/R36 B1; sequential single TP2 services on the same cards",
        repetitions=1,
        accuracy_evaluated=False,
        output_throughput_change_percent=100
        * (b1["output_tps"] / b0["output_tps"] - 1),
        mean_ttft_reduction_percent=100
        * (1 - b1["ttft_s"]["mean"] / b0["ttft_s"]["mean"]),
        mean_tpot_reduction_percent=100
        * (1 - b1["tpot_s"]["mean"] / b0["tpot_s"]["mean"]),
        resource_cost_reduction_percent=100
        * (
            1
            - b1["allocated_card_seconds_per_output_token"]
            / b0["allocated_card_seconds_per_output_token"]
        ),
        lifecycle_resource_cost_change_percent=100
        * (
            b1["lifecycle_allocated_card_seconds_per_output_token"]
            / b0["lifecycle_allocated_card_seconds_per_output_token"]
            - 1
        ),
        p99_tpot_change_percent=100 * (b1["tpot_s"]["p99"] / b0["tpot_s"]["p99"] - 1),
    )


if __name__ == "__main__":
    root = Path(sys.argv[1])
    result = compare(root, sys.argv[2], sys.argv[3])
    (root / "paired-results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
