"""Translate upstream measurements without changing their physical meaning.

Offline batch completion latency is not TTFT, requests/s is not tokens/s,
and an unmeasured memory peak is not zero. Preserve absent measurements as null.
"""

from typing import Any


def safe_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def first_measured(payload: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = safe_float(payload.get(key))
        if value is not None:
            return value
    return None


def derive_metrics_from_benchmark_result(
    benchmark_result_payload: dict[str, Any],
    *,
    peak_mem_mb: float | None,
    benchmark_type: str = "serve",
) -> dict[str, Any]:
    raw = benchmark_result_payload
    completed = int(raw.get("completed") or 0)
    failed = int(raw.get("failed") or 0)
    total = completed + failed
    errors = raw.get("errors") or []
    if total == 0 and isinstance(errors, list) and errors:
        failed = sum(1 for item in errors if item)
        total = len(errors)
    peak = (
        peak_mem_mb if peak_mem_mb is not None else safe_float(raw.get("peak_mem_mb"))
    )
    metrics = {
        "ttft_ms": safe_float(raw.get("mean_ttft_ms")),
        # The existing catalog defines tpot_ms as an alias of tbt_ms. Preserve
        # this public contract; per-token ITL distributions stay in the raw file.
        "tbt_ms": first_measured(
            raw, "mean_tpot_ms", "mean_tbt_ms", "tpot_ms", "tbt_ms"
        ),
        "throughput_tps": first_measured(
            raw, "output_throughput", "tokens_per_second", "total_token_throughput"
        ),
        "peak_mem_mb": int(peak) if peak is not None else None,
        "error_rate": failed / total if total else None,
    }
    if benchmark_type in ("throughput", "latency"):
        metrics["ttft_ms"] = None
        metrics["tbt_ms"] = None
    if benchmark_type == "latency":
        metrics["throughput_tps"] = None
        latency = safe_float(raw.get("avg_latency"))
        metrics["batch_latency_ms"] = latency * 1000 if latency is not None else None
    return metrics
