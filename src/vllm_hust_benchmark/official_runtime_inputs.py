from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Mapping


SHAREGPT_DATASET_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


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

    return dataset_path


def normalize_client_parameters(
    parameters: Mapping[str, Any],
    *,
    benchmark_type: str,
    ready_check_timeout_sec: int | None = None,
    vllm_worktree: str | None = None,
    dataset_cache_root: str | None = None,
) -> dict[str, Any]:
    normalized = dict(parameters)

    if benchmark_type == "serve":
        if ready_check_timeout_sec and int(normalized.get("ready_check_timeout_sec") or 0) <= 0:
            normalized["ready_check_timeout_sec"] = ready_check_timeout_sec
    else:
        normalized.pop("ready_check_timeout_sec", None)

    if benchmark_type == "throughput":
        # v0.11.0 throughput CLI does not accept this newer flag.
        normalized.pop("num_warmups", None)

    if "dataset_path" in normalized:
        normalized["dataset_path"] = resolve_runtime_dataset_path(
            normalized["dataset_path"],
            vllm_worktree=vllm_worktree,
            dataset_cache_root=dataset_cache_root,
        )

    return normalized