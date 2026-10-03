#!/usr/bin/env python3
"""Materialize one hash-matched fixed-target pair and its audit summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


UNKNOWN_METRICS = (
    "single_chip_effective_utilization_pct",
    "unit_token_cost_reduction_pct",
    "multi_tenant_high_utilization",
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _ratio(current: float, baseline: float) -> float:
    if baseline <= 0:
        raise ValueError("baseline metric must be positive")
    return current / baseline


def _reduction(current: float, baseline: float) -> float:
    return (1.0 - _ratio(current, baseline)) * 100.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(
    baseline_path: Path,
    current_path: Path,
    baseline_raw_path: Path,
    current_raw_path: Path,
    output_dir: Path,
) -> None:
    baseline = _load(baseline_path)
    current = _load(current_path)
    baseline_raw = _load(baseline_raw_path)
    current_raw = _load(current_raw_path)

    baseline_hash = (baseline.get("same_spec") or {}).get("resolved_spec_hash")
    current_hash = (current.get("same_spec") or {}).get("resolved_spec_hash")
    if not baseline_hash or baseline_hash != current_hash:
        raise ValueError(
            "fixed-target pair requires identical non-empty resolved_spec_hash values"
        )

    baseline_metrics = baseline["metrics"]
    current_metrics = current["metrics"]
    throughput_ratio = _ratio(
        float(current_metrics["throughput_tps"]),
        float(baseline_metrics["throughput_tps"]),
    )
    ttft_reduction = _reduction(
        float(current_metrics["ttft_ms"]), float(baseline_metrics["ttft_ms"])
    )
    tpot_reduction = _reduction(
        float(current_metrics["tbt_ms"]), float(baseline_metrics["tbt_ms"])
    )

    constraints = current.setdefault("constraints", {})
    accountable = constraints.setdefault("accountable_scope", {})
    accountable["baseline_engine"] = "vllm"
    accountable["declared_baseline_engine"] = "vllm"
    accountable["baseline_status"] = "matched-same-spec"
    metrics = constraints.setdefault("metrics", {})
    metrics["typical_throughput_ratio_vs_baseline"] = throughput_ratio
    metrics["typical_ttft_reduction_pct_vs_baseline"] = ttft_reduction
    metrics["typical_tpot_reduction_pct_vs_baseline"] = tpot_reduction

    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = output_dir / "current_run_leaderboard.json"
    enriched_path.write_text(
        json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    raw_arrays = {}
    for label, payload in (("baseline", baseline_raw), ("current", current_raw)):
        raw_arrays[label] = {
            key: len(payload.get(key) or [])
            for key in ("input_lens", "output_lens", "ttfts", "itls", "errors")
        }

    summary = {
        "schema_version": "fixed-target-pair-evidence/v1",
        "pair_status": "matched",
        "resolved_spec_hash": baseline_hash,
        "baseline": {
            "entry_id": baseline.get("entry_id"),
            "engine": baseline.get("engine"),
            "engine_version": baseline.get("engine_version"),
            "metrics": baseline_metrics,
            "p95": {
                "ttft_ms": (baseline.get("constraints") or {})
                .get("metrics", {})
                .get("long_context_ttft_p95_ms"),
                "tpot_ms": (baseline.get("constraints") or {})
                .get("metrics", {})
                .get("long_context_tpot_p95_ms"),
            },
            "raw_sha256": _sha256(baseline_raw_path),
        },
        "current": {
            "entry_id": current.get("entry_id"),
            "engine": current.get("engine"),
            "engine_version": current.get("engine_version"),
            "metrics": current_metrics,
            "p95": {
                "ttft_ms": metrics.get("long_context_ttft_p95_ms"),
                "tpot_ms": metrics.get("long_context_tpot_p95_ms"),
            },
            "raw_sha256": _sha256(current_raw_path),
        },
        "derived": {
            "throughput_ratio": {
                "value": throughput_ratio,
                "formula": "current.throughput_tps / baseline.throughput_tps",
            },
            "ttft_reduction_pct": {
                "value": ttft_reduction,
                "formula": "(1 - current.mean_ttft_ms / baseline.mean_ttft_ms) * 100",
            },
            "tpot_reduction_pct": {
                "value": tpot_reduction,
                "formula": "(1 - current.mean_tpot_ms / baseline.mean_tpot_ms) * 100",
            },
        },
        "hard_constraint_inputs": {
            "measured": {
                "typical_throughput_ratio_vs_baseline": throughput_ratio,
                "typical_ttft_reduction_pct_vs_baseline": ttft_reduction,
                "typical_tpot_reduction_pct_vs_baseline": tpot_reduction,
                "long_context_ttft_p95_ms": metrics.get(
                    "long_context_ttft_p95_ms"
                ),
                "long_context_tpot_p95_ms": metrics.get(
                    "long_context_tpot_p95_ms"
                ),
            },
            "unknown": {
                name: {
                    "value": None,
                    "status": "not-measured",
                    "reason": "the matched run did not collect the required denominator or telemetry",
                }
                for name in UNKNOWN_METRICS
            },
        },
        "raw_request_array_lengths": raw_arrays,
        "p95_source": "vllm bench serve raw request distributions; metric_percentiles=95,99",
    }
    (output_dir / "pair_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--current-raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    materialize(
        args.baseline,
        args.current,
        args.baseline_raw,
        args.current_raw,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
