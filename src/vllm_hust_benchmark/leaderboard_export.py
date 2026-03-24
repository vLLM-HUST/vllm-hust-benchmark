from __future__ import annotations

import hashlib
import json
import platform
import sys
import uuid
from datetime import UTC, datetime
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
    metrics_file: Path,
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
) -> tuple[Path, Path]:
    payload = _load_metrics_payload(metrics_file)
    metrics = dict(payload["metrics"])
    constraints_metrics = dict(payload["constraints_metrics"])

    submitted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
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
