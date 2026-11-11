"""Normalize the two observed vLLM timing schemas without mixing clock origins."""

import math
import statistics


def request_metrics(metrics, before, end, tokens):
    first = metrics.get("first_token_time")
    arrival = metrics.get("arrival_time")
    finished = metrics.get("finished_time")
    if first is not None and arrival is not None:
        ttft = 1000 * (first - arrival)
        tpot = (
            1000 * (finished - first) / (tokens - 1)
            if finished is not None and tokens > 1
            else None
        )
        return {
            "ttft_ms": ttft,
            "tpot_ms": tpot,
            "timing_basis": "engine first_token_time/arrival_time/finished_time",
        }
    first = metrics.get("first_token_ts")
    latency = metrics.get("first_token_latency")
    if latency is None or first is None:
        return {
            "ttft_ms": None,
            "tpot_ms": None,
            "timing_basis": "no usable native timing fields",
        }
    if not before <= first <= end:
        raise ValueError(
            "Engine first_token_ts is outside the monotonic request lifetime; cannot combine clocks"
        )
    # arrival_time is an epoch timestamp in this schema; first_token_ts is
    # monotonic. Use the native latency scalar for TTFT, never subtract the two.
    return {
        "ttft_ms": 1000 * latency,
        "tpot_ms": 1000 * (end - first) / (tokens - 1) if tokens > 1 else None,
        "timing_basis": "native first_token_latency; monotonic first_token_ts to generate return for TPOT",
    }


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    x = (len(ordered) - 1) * p
    low, high = math.floor(x), math.ceil(x)
    return ordered[low] + (ordered[high] - ordered[low]) * (x - low)


def group_metrics(rows):
    seconds = sum(r["elapsed_s"] for r in rows)
    result = {
        "completed": len(rows),
        "output_tokens": sum(r["output_tokens"] for r in rows),
        "generate_wall_s": seconds,
        "length_stops": sum(r["finish_reason"] == "length" for r in rows),
        "output_tps": sum(r["output_tokens"] for r in rows) / seconds
        if seconds
        else None,
    }
    for metric in ["ttft_ms", "tpot_ms"]:
        samples = [r[metric] for r in rows if r.get(metric) is not None]
        result[metric] = {
            "samples": len(samples),
            "mean": statistics.mean(samples) if samples else None,
            "p95": percentile(samples, 0.95),
            "p99": percentile(samples, 0.99),
        }
    return result
