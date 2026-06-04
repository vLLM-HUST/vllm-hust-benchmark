from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.registry import get_scenario

PRECISION_TO_DTYPE = {
    "FP32": "float32",
    "FP16": "float16",
    "BF16": "bfloat16",
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}

NON_SEMANTIC_SERVER_KEYS = {"host", "port", "model"}
NON_SEMANTIC_CLIENT_KEYS = {"host", "port", "model"}
LOCAL_MODEL_CONFIG_FILES = ("config.json",)
LOCAL_MODEL_TOKENIZER_PATTERNS = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "vocab.json",
    "vocab.txt",
)
LOCAL_MODEL_WEIGHT_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
)
LOCAL_MODEL_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
PREFIX_REPETITION_DEFAULT_SUFFIX_LEN = 256
PREFIX_REPETITION_DEFAULT_NUM_PREFIXES = 10
GPU_MEMORY_UTILIZATION_ENV = "SAME_SPEC_GPU_MEMORY_UTILIZATION"
MAX_MODEL_LEN_ENV = "SAME_SPEC_MAX_MODEL_LEN"


def _path_has_any_matching_file(path: Path, patterns: tuple[str, ...]) -> bool:
    return any(candidate.is_file() for pattern in patterns for candidate in path.glob(pattern))


def _path_has_complete_indexed_weights(path: Path) -> bool:
    for index_name in LOCAL_MODEL_INDEX_FILES:
        index_path = path / index_name
        if not index_path.is_file():
            continue

        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map") or {}
        if not isinstance(weight_map, dict) or not weight_map:
            continue

        shard_names = {str(filename) for filename in weight_map.values() if filename}
        if shard_names and all((path / shard_name).is_file() for shard_name in shard_names):
            return True

    return False


def _benchmark_type_for_spec(spec: dict[str, Any]) -> str:
    return get_scenario(_require_string(spec, "scenario")).benchmark_type


def _maybe_apply_gpu_memory_utilization_override(
    spec: dict[str, Any],
    resolved: dict[str, Any],
    *,
    parameter_set: str,
) -> None:
    if "gpu_memory_utilization" in resolved:
        return

    if parameter_set == "client" and _benchmark_type_for_spec(spec) == "serve":
        return

    override = os.environ.get(GPU_MEMORY_UTILIZATION_ENV, "").strip()
    if not override:
        return

    try:
        resolved["gpu_memory_utilization"] = float(override)
    except ValueError as exc:
        raise ValueError(
            f"{GPU_MEMORY_UTILIZATION_ENV} must be a float, got {override!r}"
        ) from exc


def _maybe_apply_max_model_len_override(
    spec: dict[str, Any],
    resolved: dict[str, Any],
    *,
    parameter_set: str,
) -> None:
    if "max_model_len" in resolved:
        return

    if parameter_set not in {"server", "client"}:
        return

    if parameter_set == "client" and _benchmark_type_for_spec(spec) == "serve":
        return

    override = os.environ.get(MAX_MODEL_LEN_ENV, "").strip()
    if not override:
        return

    try:
        max_model_len = int(override)
    except ValueError as exc:
        raise ValueError(
            f"{MAX_MODEL_LEN_ENV} must be an integer, got {override!r}"
        ) from exc

    if max_model_len <= 0:
        raise ValueError(f"{MAX_MODEL_LEN_ENV} must be > 0, got {override!r}")

    resolved["max_model_len"] = max_model_len


def runtime_model_path_has_required_artifacts(candidate: str | Path) -> bool:
    path = Path(candidate)
    if not path.is_dir():
        return False

    return (
        _path_has_any_matching_file(path, LOCAL_MODEL_CONFIG_FILES)
        and _path_has_any_matching_file(path, LOCAL_MODEL_TOKENIZER_PATTERNS)
        and (
            _path_has_any_matching_file(path, LOCAL_MODEL_WEIGHT_PATTERNS)
            or _path_has_complete_indexed_weights(path)
        )
    )


def load_benchmark_spec(spec_file: Path) -> dict[str, Any]:
    payload = json.loads(spec_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{spec_file}: benchmark spec must be a JSON object")
    return payload


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ValueError(f"benchmark spec is missing required field: {key}")
    return value


def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"benchmark spec field must be an object: {key}")
    return dict(value)


def precision_to_runtime_dtype(model_precision: str) -> str:
    dtype = PRECISION_TO_DTYPE.get(model_precision)
    if dtype is None:
        raise ValueError(
            "benchmark spec model_precision is not mappable to a runtime dtype: "
            f"{model_precision!r}"
        )
    return dtype


def resolve_server_parameters(
    spec: dict[str, Any],
    *,
    runtime_model: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    resolved = _require_dict(spec, "server_parameters")
    resolved["model"] = runtime_model or _require_string(spec, "model")
    if host is not None:
        resolved["host"] = host
    if port is not None:
        resolved["port"] = port
    _maybe_apply_gpu_memory_utilization_override(spec, resolved, parameter_set="server")
    _maybe_apply_max_model_len_override(spec, resolved, parameter_set="server")
    if "dtype" not in resolved:
        resolved["dtype"] = precision_to_runtime_dtype(_require_string(spec, "model_precision"))
    if "enforce_eager" not in resolved:
        resolved["enforce_eager"] = ""
    return resolved


def resolve_client_parameters(
    spec: dict[str, Any],
    *,
    runtime_model: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    resolved = _require_dict(spec, "client_parameters")
    resolved["model"] = runtime_model or _require_string(spec, "model")
    if host is not None:
        resolved["host"] = host
    if port is not None:
        resolved["port"] = port
    _maybe_apply_gpu_memory_utilization_override(spec, resolved, parameter_set="client")
    _maybe_apply_max_model_len_override(spec, resolved, parameter_set="client")

    if resolved.get("dataset_name") == "random":
        if "input_len" in resolved and "random_input_len" not in resolved:
            resolved["random_input_len"] = resolved.pop("input_len")
        if "output_len" in resolved and "random_output_len" not in resolved:
            resolved["random_output_len"] = resolved.pop("output_len")

    if resolved.get("dataset_name") == "prefix_repetition":
        total_input_len = resolved.pop("input_len", None)

        if "output_len" in resolved:
            if "prefix_repetition_output_len" not in resolved:
                resolved["prefix_repetition_output_len"] = resolved["output_len"]
            resolved.pop("output_len", None)

        if "prefix_repetition_num_prefixes" not in resolved:
            resolved["prefix_repetition_num_prefixes"] = PREFIX_REPETITION_DEFAULT_NUM_PREFIXES

        if total_input_len is not None:
            total_input_len = int(total_input_len)
            if "prefix_repetition_suffix_len" not in resolved:
                # Legacy specs only carry the total input length. Keep the
                # default varying suffix budget and assign the rest to the
                # shared prefix so the prompt length remains unchanged.
                resolved["prefix_repetition_suffix_len"] = min(
                    PREFIX_REPETITION_DEFAULT_SUFFIX_LEN,
                    total_input_len,
                )
            if "prefix_repetition_prefix_len" not in resolved:
                resolved["prefix_repetition_prefix_len"] = max(
                    total_input_len - int(resolved["prefix_repetition_suffix_len"]),
                    0,
                )
    return resolved


def _normalize_for_hash(
    parameters: dict[str, Any], *, drop_keys: set[str]
) -> dict[str, Any]:
    return {key: value for key, value in parameters.items() if key not in drop_keys}


def build_same_spec_payload(
    spec: dict[str, Any],
    *,
    spec_source: Path | None = None,
    runtime_model: str | None = None,
    server_host: str | None = None,
    server_port: int | None = None,
    client_host: str | None = None,
    client_port: int | None = None,
) -> dict[str, Any]:
    spec_id = _require_string(spec, "id")
    canonical_model = _require_string(spec, "model")
    resolved_server_parameters = resolve_server_parameters(
        spec,
        runtime_model=runtime_model or canonical_model,
        host=server_host,
        port=server_port,
    )
    resolved_client_parameters = resolve_client_parameters(
        spec,
        runtime_model=runtime_model or canonical_model,
        host=client_host,
        port=client_port,
    )
    hash_basis = {
        "schema_version": "benchmark-same-spec/v1",
        "spec_id": spec_id,
        "scenario": _require_string(spec, "scenario"),
        "model": canonical_model,
        "model_parameters": _require_string(spec, "model_parameters"),
        "model_precision": _require_string(spec, "model_precision"),
        "hardware_vendor": _require_string(spec, "hardware_vendor"),
        "hardware_chip_model": _require_string(spec, "hardware_chip_model"),
        "chip_count": int(spec.get("chip_count") or 0),
        "node_count": int(spec.get("node_count") or 0),
        "resolved_server_parameters": _normalize_for_hash(
            resolve_server_parameters(
                spec,
                runtime_model=canonical_model,
                host=server_host,
                port=server_port,
            ),
            drop_keys=NON_SEMANTIC_SERVER_KEYS,
        ),
        "resolved_client_parameters": _normalize_for_hash(
            resolve_client_parameters(
                spec,
                runtime_model=canonical_model,
                host=client_host,
                port=client_port,
            ),
            drop_keys=NON_SEMANTIC_CLIENT_KEYS,
        ),
    }
    hash_input = json.dumps(
        hash_basis,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    resolved_spec_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    return {
        "schema_version": "benchmark-same-spec/v1",
        "spec_id": spec_id,
        "spec_label": str(spec.get("label") or ""),
        "spec_source": str(spec_source.resolve()) if spec_source is not None else None,
        "scenario": _require_string(spec, "scenario"),
        "model": canonical_model,
        "model_parameters": _require_string(spec, "model_parameters"),
        "model_precision": _require_string(spec, "model_precision"),
        "hardware_vendor": _require_string(spec, "hardware_vendor"),
        "hardware_chip_model": _require_string(spec, "hardware_chip_model"),
        "chip_count": int(spec.get("chip_count") or 0),
        "node_count": int(spec.get("node_count") or 0),
        "resolved_spec_hash": resolved_spec_hash,
        "resolved_server_parameters": resolved_server_parameters,
        "resolved_client_parameters": resolved_client_parameters,
    }


def write_same_spec_payload(
    *,
    spec_file: Path,
    output_file: Path,
    runtime_model: str | None = None,
    server_host: str | None = None,
    server_port: int | None = None,
    client_host: str | None = None,
    client_port: int | None = None,
) -> Path:
    payload = build_same_spec_payload(
        load_benchmark_spec(spec_file),
        spec_source=spec_file,
        runtime_model=runtime_model,
        server_host=server_host,
        server_port=server_port,
        client_host=client_host,
        client_port=client_port,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve a benchmark same-spec payload for runtime and export.",
    )
    parser.add_argument("resolve", nargs="?")
    parser.add_argument("--spec-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--runtime-model")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve a benchmark same-spec payload for runtime and export.",
    )
    parser.add_argument("command", choices=["resolve"])
    parser.add_argument("--spec-file", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--runtime-model")
    parser.add_argument("--server-host")
    parser.add_argument("--server-port", type=int)
    parser.add_argument("--client-host")
    parser.add_argument("--client-port", type=int)
    args = parser.parse_args(argv)

    try:
        output_file = write_same_spec_payload(
            spec_file=args.spec_file.resolve(),
            output_file=args.output_file.resolve(),
            runtime_model=args.runtime_model,
            server_host=args.server_host,
            server_port=args.server_port,
            client_host=args.client_host,
            client_port=args.client_port,
        )
    except (OSError, ValueError) as error:
        print(str(error))
        return 2

    print(output_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())