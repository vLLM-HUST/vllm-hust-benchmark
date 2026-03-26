from __future__ import annotations

import hashlib
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm_hust_benchmark import __version__ as benchmark_version
from vllm_hust_benchmark.models import ScenarioDefinition

REQUIRED_METRIC_KEYS = (
    "ttft_ms",
    "throughput_tps",
    "peak_mem_mb",
    "error_rate",
)

REQUIRED_CONSTRAINT_METRIC_KEYS = (
    "single_chip_effective_utilization_pct",
    "typical_throughput_ratio_vs_baseline",
    "typical_ttft_reduction_pct_vs_baseline",
    "typical_tpot_reduction_pct_vs_baseline",
    "long_context_length",
    "long_context_throughput_stable",
    "long_context_ttft_p95_ms",
    "long_context_ttft_p99_ms",
    "long_context_tpot_p95_ms",
    "long_context_tpot_p99_ms",
    "long_context_ttft_p95_stable",
    "long_context_ttft_p99_stable",
    "long_context_tpot_p95_stable",
    "long_context_tpot_p99_stable",
    "unit_token_cost_reduction_pct",
    "multi_tenant_high_utilization",
)


def _load_metrics_payload(metrics_file: Path) -> dict[str, Any]:
    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metrics file must be a JSON object")

    metrics = payload.get("metrics")
    constraints_metrics = payload.get("constraints_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("metrics file must include object key: metrics")
    if not isinstance(constraints_metrics, dict):
        raise ValueError("metrics file must include object key: constraints_metrics")

    missing_metrics = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing_metrics:
        raise ValueError(f"metrics missing required keys: {', '.join(missing_metrics)}")

    missing_constraints = [
        key for key in REQUIRED_CONSTRAINT_METRIC_KEYS if key not in constraints_metrics
    ]
    if missing_constraints:
        raise ValueError(
            "constraints_metrics missing required keys: "
            + ", ".join(missing_constraints)
        )

    return payload


def _load_constraints_metrics(constraints_file: Path) -> dict[str, Any]:
    payload = json.loads(constraints_file.read_text(encoding="utf-8"))
    constraints_metrics = payload.get("constraints_metrics", payload)
    if not isinstance(constraints_metrics, dict):
        raise ValueError("constraints file must be a JSON object or include constraints_metrics")

    missing_constraints = [
        key for key in REQUIRED_CONSTRAINT_METRIC_KEYS if key not in constraints_metrics
    ]
    if missing_constraints:
        raise ValueError(
            "constraints_metrics missing required keys: " + ", ".join(missing_constraints)
        )
    return dict(constraints_metrics)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _derive_metrics_from_benchmark_result(
    benchmark_result_file: Path,
    *,
    peak_mem_mb: float | None,
) -> dict[str, Any]:
    payload = json.loads(benchmark_result_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark result file must be a JSON object")

    completed = int(payload.get("completed") or 0)
    failed = int(payload.get("failed") or 0)
    total = completed + failed
    errors = payload.get("errors") or []
    if total == 0 and isinstance(errors, list) and errors:
        failed = sum(1 for item in errors if item)
        completed = len(errors) - failed
        total = len(errors)

    mean_ttft_ms = _safe_float(payload.get("mean_ttft_ms"))
    if mean_ttft_ms is None:
        avg_latency_seconds = _safe_float(payload.get("avg_latency"))
        if avg_latency_seconds is not None:
            mean_ttft_ms = avg_latency_seconds * 1000.0

    throughput_tps = (
        _safe_float(payload.get("output_throughput"))
        or _safe_float(payload.get("tokens_per_second"))
        or _safe_float(payload.get("total_token_throughput"))
        or _safe_float(payload.get("requests_per_second"))
        or _safe_float(payload.get("request_throughput"))
        or 0.0
    )

    if total > 0:
        error_rate = failed / total
    else:
        error_rate = 0.0

    metrics = {
        "ttft_ms": float(mean_ttft_ms or 0.0),
        "throughput_tps": float(throughput_tps),
        "peak_mem_mb": float(peak_mem_mb or payload.get("peak_mem_mb") or 0.0),
        "error_rate": float(error_rate),
    }

    missing_metrics = [key for key in REQUIRED_METRIC_KEYS if key not in metrics]
    if missing_metrics:
        raise ValueError(f"derived metrics missing required keys: {', '.join(missing_metrics)}")
    return metrics


def load_export_payload(
    *,
    metrics_file: Path | None,
    benchmark_result_file: Path | None,
    constraints_file: Path | None,
    peak_mem_mb: float | None,
) -> dict[str, Any]:
    if metrics_file is not None:
        return _load_metrics_payload(metrics_file)
    if benchmark_result_file is None or constraints_file is None:
        raise ValueError(
            "provide either metrics_file, or benchmark_result_file together with constraints_file"
        )
    return {
        "metrics": _derive_metrics_from_benchmark_result(
            benchmark_result_file,
            peak_mem_mb=peak_mem_mb,
        ),
        "constraints_metrics": _load_constraints_metrics(constraints_file),
    }


def _infer_config_type(
    *, chip_count: int, node_count: int, scenario: ScenarioDefinition
) -> str:
    if node_count > 1:
        return "multi_node"
    if chip_count > 1:
        return "multi_gpu"
    return str(scenario.leaderboard.get("default_config_type") or "single_gpu")


def _infer_workload_lengths(
    scenario: ScenarioDefinition, input_length: int | None, output_length: int | None
) -> tuple[int, int]:
    inferred_input = input_length or int(scenario.defaults.get("input_len") or 1024)
    inferred_output = output_length or int(scenario.defaults.get("output_len") or 256)
    return inferred_input, inferred_output


def _build_idempotency_key(
    *,
    scenario_name: str,
    engine: str,
    engine_version: str,
    model_name: str,
    hardware_chip_model: str,
    chip_count: int,
    node_count: int,
    run_id: str,
) -> str:
    raw = "|".join(
        [
            scenario_name,
            engine,
            engine_version,
            model_name,
            hardware_chip_model,
            str(chip_count),
            str(node_count),
            run_id,
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def export_leaderboard_artifacts(
    *,
    scenario: ScenarioDefinition,
    metrics_file: Path | None,
    benchmark_result_file: Path | None,
    constraints_file: Path | None,
    output_dir: Path,
    artifact_name: str,
    run_id: str,
    engine: str,
    engine_version: str,
    model_name: str,
    model_parameters: str,
    model_precision: str,
    hardware_vendor: str,
    hardware_chip_model: str,
    chip_count: int,
    node_count: int,
    submitter: str,
    baseline_engine: str,
    domestic_chip_class: str,
    representative_model_band: str,
    data_source: str,
    input_length: int | None,
    output_length: int | None,
    batch_size: int | None,
    concurrent_requests: int | None,
    protocol_version: str,
    backend_version: str,
    core_version: str,
    peak_mem_mb: float | None,
) -> tuple[Path, Path]:
    payload = load_export_payload(
        metrics_file=metrics_file,
        benchmark_result_file=benchmark_result_file,
        constraints_file=constraints_file,
        peak_mem_mb=peak_mem_mb,
    )
    metrics = dict(payload["metrics"])
    constraints_metrics = dict(payload["constraints_metrics"])

    submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    config_type = _infer_config_type(
        chip_count=chip_count, node_count=node_count, scenario=scenario
    )
    workload_input, workload_output = _infer_workload_lengths(
        scenario, input_length, output_length
    )
    workload_name = str(scenario.leaderboard.get("workload_name") or scenario.name)
    representative_business_scenario = str(
        scenario.leaderboard.get("representative_business_scenario") or "general-serving"
    )
    idempotency_key = _build_idempotency_key(
        scenario_name=scenario.name,
        engine=engine,
        engine_version=engine_version,
        model_name=model_name,
        hardware_chip_model=hardware_chip_model,
        chip_count=chip_count,
        node_count=node_count,
        run_id=run_id,
    )

    cluster: dict[str, Any] | None
    if node_count > 1:
        cluster = {
            "node_count": node_count,
            "comm_backend": "unknown",
            "topology_type": "unknown",
        }
    else:
        cluster = None

    dataset_name = scenario.defaults.get("dataset_name")
    if dataset_name is None:
        dataset_name = scenario.defaults.get("dataset_path")

    artifact = {
        "entry_id": str(uuid.uuid4()),
        "engine": engine,
        "engine_version": engine_version,
        "config_type": config_type,
        "hardware": {
            "vendor": hardware_vendor,
            "chip_model": hardware_chip_model,
            "chip_count": chip_count,
            "interconnect": "unknown",
        },
        "model": {
            "name": model_name,
            "parameters": model_parameters,
            "precision": model_precision,
            "quantization": None,
        },
        "workload": {
            "name": workload_name,
            "input_length": workload_input,
            "output_length": workload_output,
            "batch_size": batch_size,
            "concurrent_requests": concurrent_requests,
            "dataset": dataset_name,
        },
        "metrics": metrics,
        "constraints": {
            "scenario_source": "vllm-benchmark",
            "accountable_scope": {
                "domestic_chip_class": domestic_chip_class,
                "representative_model_band": representative_model_band,
                "representative_business_scenario": representative_business_scenario,
                "baseline_engine": baseline_engine,
                "owner_confirmed": None,
                "notes": None,
            },
            "metrics": constraints_metrics,
        },
        "cluster": cluster,
        "versions": {
            "protocol": protocol_version,
            "backend": backend_version,
            "core": core_version,
            "benchmark": benchmark_version,
        },
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": None,
            "cuda_version": None,
            "cann_version": None,
            "driver_version": None,
        },
        "metadata": {
            "submitted_at": submitted_at,
            "submitter": submitter,
            "data_source": data_source,
            "engine": engine,
            "engine_version": engine_version,
            "reproducible_cmd": None,
            "git_commit": None,
            "release_date": None,
            "changelog_url": None,
            "notes": None,
            "verified": None,
            "idempotency_key": idempotency_key,
            "manifest_source": f"generated-by-vllm-hust-benchmark/{benchmark_version}",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / artifact_name
    artifact_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest_path = output_dir / "leaderboard_manifest.json"
    manifest = {
        "schema_version": "leaderboard-export-manifest/v2",
        "generated_at": submitted_at,
        "entries": [
            {
                "idempotency_key": idempotency_key,
                "leaderboard_artifact": artifact_name,
            }
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return artifact_path, manifest_path
