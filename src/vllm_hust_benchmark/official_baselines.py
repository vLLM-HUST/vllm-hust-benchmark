from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Mapping

from vllm_hust_benchmark.registry import get_scenario

OFFICIAL_BASELINE_SUBMITTER = "official-ascend-baseline"

PRIMARY_METRIC_BY_BENCHMARK_TYPE = {
    "serve": "ttft_ms",
    "latency": "ttft_ms",
    "throughput": "throughput_tps",
}


def load_official_baseline_spec(spec_file: Path) -> dict[str, Any]:
    payload = json.loads(spec_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{spec_file}: official baseline spec must be a JSON object")
    return payload


def get_official_baseline_spec_id(spec: Mapping[str, Any]) -> str:
    spec_id = str(spec.get("id") or "").strip()
    if not spec_id:
        raise ValueError("official baseline spec is missing required field: id")
    return spec_id


def get_canonical_submission_dir(
    spec: Mapping[str, Any], *, submissions_root: Path
) -> Path:
    return submissions_root / get_official_baseline_spec_id(spec)


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def has_canonical_run(spec: Mapping[str, Any], *, submissions_root: Path) -> bool:
    canonical_dir = get_canonical_submission_dir(spec, submissions_root=submissions_root)
    run_file = canonical_dir / "run_leaderboard.json"
    manifest_file = canonical_dir / "leaderboard_manifest.json"

    run_payload = _load_json_object(run_file)
    manifest_payload = _load_json_object(manifest_file)
    if run_payload is None or manifest_payload is None:
        return False

    same_spec = run_payload.get("same_spec")
    same_spec = same_spec if isinstance(same_spec, Mapping) else {}
    metadata = run_payload.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    entries = manifest_payload.get("entries")
    entries = entries if isinstance(entries, list) else []

    spec_id = get_official_baseline_spec_id(spec)
    if str(same_spec.get("spec_id") or "").strip() != spec_id:
        return False
    if str(metadata.get("submitter") or "").strip() != OFFICIAL_BASELINE_SUBMITTER:
        return False
    if not any(
        isinstance(entry, Mapping)
        and str(entry.get("leaderboard_artifact") or "").strip() == "run_leaderboard.json"
        for entry in entries
    ):
        return False
    return True


def get_official_baseline_benchmark_type(spec: Mapping[str, Any]) -> str:
    scenario_name = str(spec.get("scenario") or "").strip()
    if not scenario_name:
        raise ValueError("official baseline spec is missing required field: scenario")
    return get_scenario(scenario_name).benchmark_type


def get_primary_metric_name_for_benchmark_type(benchmark_type: str) -> str:
    try:
        return PRIMARY_METRIC_BY_BENCHMARK_TYPE[benchmark_type]
    except KeyError as exc:
        raise ValueError(
            f"unsupported official baseline benchmark type: {benchmark_type}"
        ) from exc


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_result_artifact_payload(result_dir: Path) -> dict[str, Any] | None:
    return _load_json_object(result_dir / "submission" / "run_leaderboard.json")


def select_canonical_candidate(
    result_dirs: list[Path], *, benchmark_type: str
) -> dict[str, Any]:
    primary_metric_name = get_primary_metric_name_for_benchmark_type(benchmark_type)

    candidates: list[dict[str, Any]] = []
    for index, result_dir in enumerate(result_dirs):
        payload = _load_result_artifact_payload(result_dir)
        if payload is None:
            continue

        metrics = payload.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        primary_metric_value = _safe_float(metrics.get(primary_metric_name))
        if primary_metric_value is None:
            continue

        error_rate = _safe_float(metrics.get("error_rate"))
        candidates.append(
            {
                "result_dir": str(result_dir.resolve()),
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
                "error_rate": float(error_rate or 0.0),
                "index": index,
            }
        )

    if not candidates:
        raise ValueError("no valid repeated runs available for canonical candidate selection")

    metric_median = median(item["primary_metric_value"] for item in candidates)
    for candidate in candidates:
        candidate["distance_to_median"] = abs(
            candidate["primary_metric_value"] - metric_median
        )

    selected = min(
        candidates,
        key=lambda item: (
            item["error_rate"],
            item["distance_to_median"],
            item["index"],
        ),
    )

    return {
        "benchmark_type": benchmark_type,
        "primary_metric_name": primary_metric_name,
        "median_value": float(metric_median),
        "selected_result_dir": selected["result_dir"],
        "candidates": candidates,
    }