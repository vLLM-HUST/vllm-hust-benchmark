"""Perfgate measurement strategy: warmup + representative measured run.

This module implements the P0-7 fix: instead of producing a perfgate baseline
from a single cold run, the producer executes ``warmup_runs`` discarded warmup
runs followed by ``measured_runs`` measured runs against the same live server.
The measured runs are sorted by throughput and then run index; the run in the
middle position is the representative result. An odd measured-run count is
required so that the middle position is unambiguous.

The published client metrics all come from that one real run. ``error_rate``
must be exactly 0 in every measured run. ``peak_mem_mb`` is a server-lifetime
property measured across the whole session, so the template value is kept.

The full strategy and every measured run's raw metrics are recorded in a
``measurement`` block so consumers and auditors can verify the selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

MEASUREMENT_SCHEMA_VERSION = "perfgate-measurement/v2"
MEASUREMENT_STRATEGY = "warmup+primary-median-run"
PRIMARY_METRIC = "throughput_tps"
SORT_DIRECTION = "ascending"
SECONDARY_SORT_KEY = "run_index"
SUPPORTED_AGGREGATIONS = frozenset({"primary-median-run"})
SELECTED_RUN_METRICS = ("throughput_tps", "ttft_ms", "tbt_ms", "error_rate")
PERFORMANCE_METRICS = ("throughput_tps", "ttft_ms", "tbt_ms")
PER_RUN_METRICS = ("throughput_tps", "ttft_ms", "tbt_ms", "error_rate", "peak_mem_mb")
MIN_PUBLICATION_WARMUP_RUNS = 1
MIN_PUBLICATION_MEASURED_RUNS = 3


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON object from {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_finite_non_negative(name: str, value: Any, *, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{context}: metric {name} is not a number: {value!r}")
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{context}: metric {name} is invalid: {number!r}")
    return number


def run_metrics_from_raw_result(path: Path) -> dict[str, Any]:
    """Derive per-run leaderboard metrics from one raw vllm-bench result file."""

    from vllm_hust_benchmark.leaderboard_export import (
        _derive_metrics_from_benchmark_result,
    )

    payload = _load_json_object(path)
    derived = _derive_metrics_from_benchmark_result(payload, peak_mem_mb=None)
    context = str(path)
    metrics: dict[str, Any] = {}
    for name in PERFORMANCE_METRICS:
        metrics[name] = _require_finite_non_negative(
            name, derived.get(name), context=context
        )
    error_rate = _require_finite_non_negative(
        "error_rate", derived.get("error_rate"), context=context
    )
    if error_rate != 0:
        raise ValueError(
            f"{context}: measured run has non-zero error_rate: {error_rate}"
        )
    metrics["error_rate"] = 0.0
    peak = derived.get("peak_mem_mb")
    metrics["peak_mem_mb"] = (
        _require_finite_non_negative("peak_mem_mb", peak, context=context)
        if peak is not None
        else None
    )
    return metrics


def aggregate_measured_runs(
    per_run: list[dict[str, Any]],
    *,
    warmup_runs: int,
    warmup: list[dict[str, Any]] | None = None,
    aggregation: str = "primary-median-run",
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select a representative measured run and return its client metrics.

    ``per_run`` entries must contain ``run_index``, ``raw_result_sha256`` and a
    ``metrics`` dict with the PER_RUN_METRICS keys.
    """

    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"unsupported aggregation: {aggregation!r}")
    if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int):
        raise ValueError(f"warmup_runs must be an integer, got {warmup_runs!r}")
    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0, got {warmup_runs}")
    warmup = [] if warmup is None else warmup
    if not isinstance(warmup, list) or len(warmup) != warmup_runs:
        raise ValueError(
            "warmup evidence must be a list of length warmup_runs "
            f"({warmup_runs}), got {warmup!r}"
        )
    for index, entry in enumerate(warmup, start=1):
        context = f"warmup run {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{context}: entry must be an object")
        run_index = entry.get("run_index")
        if isinstance(run_index, bool) or not isinstance(run_index, int):
            raise ValueError(f"{context}: run_index must be an integer")
        if run_index != index:
            raise ValueError(
                f"{context}: run_index mismatch: {entry.get('run_index')!r}"
            )
        digest = str(entry.get("raw_result_sha256") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                f"{context}: raw_result_sha256 must be a lowercase SHA-256 digest"
            )
    if not per_run:
        raise ValueError("at least one measured run is required")
    if len(per_run) % 2 == 0:
        raise ValueError(
            "primary-median-run requires an odd number of measured runs, "
            f"got {len(per_run)}"
        )

    for index, entry in enumerate(per_run, start=1):
        context = f"measured run {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{context}: per_run entry must be an object")
        run_index = entry.get("run_index")
        if isinstance(run_index, bool) or not isinstance(run_index, int):
            raise ValueError(f"{context}: run_index must be an integer")
        if run_index != index:
            raise ValueError(
                f"{context}: run_index mismatch: {entry.get('run_index')!r}"
            )
        digest = str(entry.get("raw_result_sha256") or "").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                f"{context}: raw_result_sha256 must be a lowercase SHA-256 digest"
            )
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"{context}: missing metrics object")
        for name in PERFORMANCE_METRICS:
            _require_finite_non_negative(name, metrics.get(name), context=context)
        error_rate = _require_finite_non_negative(
            "error_rate", metrics.get("error_rate"), context=context
        )
        if error_rate != 0:
            raise ValueError(f"{context}: non-zero error_rate: {error_rate}")
        peak_mem_mb = metrics.get("peak_mem_mb")
        if peak_mem_mb is not None:
            _require_finite_non_negative("peak_mem_mb", peak_mem_mb, context=context)

    recorded_per_run = [
        {
            "run_index": int(entry["run_index"]),
            "raw_result_sha256": str(entry["raw_result_sha256"]),
            "metrics": {
                name: (
                    float(entry["metrics"][name])
                    if entry["metrics"].get(name) is not None
                    else None
                )
                for name in PER_RUN_METRICS
            },
        }
        for entry in per_run
    ]
    ordered_runs = sorted(
        recorded_per_run,
        key=lambda entry: (
            float(entry["metrics"][PRIMARY_METRIC]),
            int(entry["run_index"]),
        ),
    )
    selected_position = len(ordered_runs) // 2 + 1
    selected = ordered_runs[selected_position - 1]
    selected_metrics = {
        name: float(selected["metrics"][name]) for name in SELECTED_RUN_METRICS
    }

    measurement = {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "strategy": MEASUREMENT_STRATEGY,
        "warmup_runs": int(warmup_runs),
        "measured_runs": len(per_run),
        "aggregation": aggregation,
        "selection": {
            "primary_metric": PRIMARY_METRIC,
            "sort_direction": SORT_DIRECTION,
            "secondary_sort_key": SECONDARY_SORT_KEY,
            "ordered_run_indices": [int(entry["run_index"]) for entry in ordered_runs],
            "selected_position": selected_position,
            "selected_run_index": int(selected["run_index"]),
            "selected_raw_result_sha256": str(selected["raw_result_sha256"]),
        },
        "warmup": [
            {
                "run_index": int(entry["run_index"]),
                "raw_result_sha256": str(entry["raw_result_sha256"]),
            }
            for entry in warmup
        ],
        "per_run": recorded_per_run,
    }
    return selected_metrics, measurement


def validate_measurement_block(
    measurement: Any,
    *,
    artifact_metrics: dict[str, Any] | None = None,
    context: str = "measurement",
    require_publication_policy: bool = False,
) -> dict[str, Any]:
    """Validate the shape of a measurement block.

    When ``artifact_metrics`` is provided, additionally cross-check that all
    published client metrics equal the selected real run.
    """

    if not isinstance(measurement, dict):
        raise ValueError(f"{context}: must be an object")
    if measurement.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
        raise ValueError(
            f"{context}: unsupported schema_version: "
            f"{measurement.get('schema_version')!r}"
        )
    if measurement.get("strategy") != MEASUREMENT_STRATEGY:
        raise ValueError(
            f"{context}: unsupported strategy: {measurement.get('strategy')!r}"
        )
    aggregation = measurement.get("aggregation")
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(f"{context}: unsupported aggregation: {aggregation!r}")
    warmup_runs = measurement.get("warmup_runs")
    if (
        isinstance(warmup_runs, bool)
        or not isinstance(warmup_runs, int)
        or warmup_runs < 0
    ):
        raise ValueError(f"{context}: invalid warmup_runs: {warmup_runs!r}")
    measured_runs = measurement.get("measured_runs")
    if (
        isinstance(measured_runs, bool)
        or not isinstance(measured_runs, int)
        or measured_runs < 1
    ):
        raise ValueError(f"{context}: invalid measured_runs: {measured_runs!r}")
    if require_publication_policy and (
        warmup_runs < MIN_PUBLICATION_WARMUP_RUNS
        or measured_runs < MIN_PUBLICATION_MEASURED_RUNS
    ):
        raise ValueError(
            f"{context}: publication requires at least "
            f"{MIN_PUBLICATION_WARMUP_RUNS} warmup run(s) and "
            f"{MIN_PUBLICATION_MEASURED_RUNS} measured run(s); got "
            f"{warmup_runs} warmup and {measured_runs} measured"
        )
    warmup = measurement.get("warmup")
    per_run = measurement.get("per_run")
    if not isinstance(per_run, list) or len(per_run) != measured_runs:
        raise ValueError(
            f"{context}: per_run must be a list of length measured_runs "
            f"({measured_runs}), got {per_run!r}"
        )
    selected_metrics, expected_measurement = aggregate_measured_runs(
        per_run,
        warmup_runs=warmup_runs,
        warmup=warmup,
        aggregation=aggregation,
    )
    selection = measurement.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"{context}: selection must be an object")
    if selection != expected_measurement["selection"]:
        raise ValueError(
            f"{context}: selection metadata does not match the recorded runs"
        )
    if artifact_metrics is not None:
        for name in SELECTED_RUN_METRICS:
            expected = selected_metrics[name]
            actual = _require_finite_non_negative(
                name, artifact_metrics.get(name), context=f"{context}: artifact"
            )
            if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"{context}: artifact metric {name}={actual!r} does not match "
                    f"selected run metric {expected!r}"
                )
    return measurement


def apply_selected_run_metrics(
    template_artifact: Path,
    selected_metrics: dict[str, float],
    output: Path,
) -> dict[str, Any]:
    """Patch the template artifact with one selected run's client metrics.

    Only client metrics are replaced; same_spec, metadata, provenance and
    peak_mem_mb are preserved from the template.
    """

    payload = _load_json_object(template_artifact)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"{template_artifact}: missing object key metrics")
    for name in SELECTED_RUN_METRICS:
        metrics[name] = float(selected_metrics[name])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def add_aggregate_runs_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--template",
        required=True,
        help="run_leaderboard.json exported from measured run 1.",
    )
    parser.add_argument(
        "--warmup-raw-result",
        action="append",
        default=[],
        dest="warmup_raw_results",
        help="Raw benchmark result file of one warmup run, in run order.",
    )
    parser.add_argument(
        "--run-raw-result",
        action="append",
        required=True,
        dest="run_raw_results",
        help="Raw benchmark result file of one measured run, in run order.",
    )
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--aggregation", default="primary-median-run")
    parser.add_argument(
        "--output",
        required=True,
        help="Path of the selected-run run_leaderboard.json.",
    )
    parser.add_argument(
        "--measurement-out",
        required=True,
        help="Path of the measurement.json strategy record.",
    )


def run_aggregate_runs_cli(args: argparse.Namespace) -> int:
    warmup = [
        {
            "run_index": index,
            "raw_result_sha256": _sha256(Path(raw)),
        }
        for index, raw in enumerate(args.warmup_raw_results, start=1)
    ]
    per_run = []
    for index, raw in enumerate(args.run_raw_results, start=1):
        raw_path = Path(raw)
        per_run.append(
            {
                "run_index": index,
                "raw_result_sha256": _sha256(raw_path),
                "metrics": run_metrics_from_raw_result(raw_path),
            }
        )
    selected_metrics, measurement = aggregate_measured_runs(
        per_run,
        warmup_runs=args.warmup_runs,
        warmup=warmup,
        aggregation=args.aggregation,
    )
    payload = apply_selected_run_metrics(
        Path(args.template), selected_metrics, Path(args.output)
    )
    validate_measurement_block(
        measurement, artifact_metrics=payload.get("metrics"), context=str(args.output)
    )
    measurement_out = Path(args.measurement_out)
    measurement_out.parent.mkdir(parents=True, exist_ok=True)
    measurement_out.write_text(
        json.dumps(measurement, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "selected measured run {selected} of {count} with {aggregation}: "
        "{output}".format(
            selected=measurement["selection"]["selected_run_index"],
            count=len(per_run),
            aggregation=args.aggregation,
            output=args.output,
        )
    )
    return 0
