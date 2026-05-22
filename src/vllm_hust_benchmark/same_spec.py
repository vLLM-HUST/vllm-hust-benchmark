from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

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
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _path_has_any_matching_file(path: Path, patterns: tuple[str, ...]) -> bool:
    return any(candidate.is_file() for pattern in patterns for candidate in path.glob(pattern))


def runtime_model_path_has_required_artifacts(candidate: str | Path) -> bool:
    path = Path(candidate)
    if not path.is_dir():
        return False

    return (
        _path_has_any_matching_file(path, LOCAL_MODEL_CONFIG_FILES)
        and _path_has_any_matching_file(path, LOCAL_MODEL_TOKENIZER_PATTERNS)
        and _path_has_any_matching_file(path, LOCAL_MODEL_WEIGHT_PATTERNS)
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

    if resolved.get("dataset_name") == "random":
        if "input_len" in resolved and "random_input_len" not in resolved:
            resolved["random_input_len"] = resolved.pop("input_len")
        if "output_len" in resolved and "random_output_len" not in resolved:
            resolved["random_output_len"] = resolved.pop("output_len")
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