from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

SHAREGPT_DATASET_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"
OFFLINE_SERVER_PARAMETER_KEYS = {
    "dtype",
    "enforce_eager",
    "gpu_memory_utilization",
    "limit_mm_per_prompt",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "model",
    "quantization",
    "tensor_parallel_size",
    "tokenizer",
    "trust_remote_code",
}


def _parse_limit_mm_per_prompt(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    mapping: dict[str, int] = {}
    for item in value.split(","):
        key, separator, raw_value = item.partition("=")
        key = key.strip()
        raw_value = raw_value.strip()
        if separator != "=" or not key or not raw_value:
            return value
        try:
            mapping[key] = int(raw_value)
        except ValueError:
            return value

    return mapping or value


def normalize_server_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(parameters)
    if "limit_mm_per_prompt" in normalized:
        normalized["limit_mm_per_prompt"] = _parse_limit_mm_per_prompt(
            normalized["limit_mm_per_prompt"]
        )
    return normalized


def resolve_runtime_dataset_path(
    dataset_path: Any,
    *,
    vllm_worktree: str | None = None,
    benchmark_repo: str | None = None,
    dataset_cache_root: str | None = None,
) -> Any:
    if not isinstance(dataset_path, str) or not dataset_path.strip():
        return dataset_path

    candidate = Path(dataset_path)
    if candidate.is_absolute():
        return str(candidate)

    if dataset_path == SHAREGPT_DATASET_FILENAME and dataset_cache_root:
        cached_sharegpt = Path(dataset_cache_root) / dataset_path
        if cached_sharegpt.is_file():
            return str(cached_sharegpt)

    if vllm_worktree:
        worktree_candidate = Path(vllm_worktree) / dataset_path
        if worktree_candidate.is_file():
            return str(worktree_candidate)

    if benchmark_repo:
        benchmark_candidate = Path(benchmark_repo) / dataset_path
        if benchmark_candidate.is_file():
            return str(benchmark_candidate)

    return dataset_path


def normalize_client_parameters(
    parameters: Mapping[str, Any],
    *,
    benchmark_type: str,
    ready_check_timeout_sec: int | None = None,
    vllm_worktree: str | None = None,
    benchmark_repo: str | None = None,
    dataset_cache_root: str | None = None,
    force_eager: bool = False,
) -> dict[str, Any]:
    normalized = dict(parameters)

    if benchmark_type == "serve":
        if (
            ready_check_timeout_sec
            and int(normalized.get("ready_check_timeout_sec") or 0) <= 0
        ):
            normalized["ready_check_timeout_sec"] = ready_check_timeout_sec
    else:
        normalized.pop("ready_check_timeout_sec", None)
        normalized.pop("temperature", None)

    if benchmark_type == "throughput":
        # v0.11.0 throughput CLI does not accept this newer flag.
        normalized.pop("num_warmups", None)

    if force_eager and benchmark_type in {"throughput", "latency"}:
        normalized["enforce_eager"] = ""

    if "dataset_path" in normalized:
        normalized["dataset_path"] = resolve_runtime_dataset_path(
            normalized["dataset_path"],
            vllm_worktree=vllm_worktree,
            benchmark_repo=benchmark_repo,
            dataset_cache_root=dataset_cache_root,
        )

    return normalized


def normalize_offline_benchmark_parameters(
    client_parameters: Mapping[str, Any],
    server_parameters: Mapping[str, Any],
    *,
    benchmark_type: str,
    ready_check_timeout_sec: int | None = None,
    vllm_worktree: str | None = None,
    benchmark_repo: str | None = None,
    dataset_cache_root: str | None = None,
    force_eager: bool = False,
) -> dict[str, Any]:
    """Build parameters for offline vllm bench throughput/latency runs.

    Throughput and latency benchmarks instantiate the engine in-process, so
    model/runtime knobs from same-spec server parameters must be carried into
    the benchmark CLI. Otherwise the artifact can say FP16 while the real
    engine silently falls back to its default dtype.
    """

    normalized = normalize_client_parameters(
        client_parameters,
        benchmark_type=benchmark_type,
        ready_check_timeout_sec=ready_check_timeout_sec,
        vllm_worktree=vllm_worktree,
        benchmark_repo=benchmark_repo,
        dataset_cache_root=dataset_cache_root,
        force_eager=force_eager,
    )
    normalized_server = normalize_server_parameters(server_parameters)

    for key in OFFLINE_SERVER_PARAMETER_KEYS:
        value = normalized_server.get(key)
        if value is None:
            continue
        if key == "model":
            normalized.setdefault(key, value)
        else:
            normalized[key] = value

    return normalized
