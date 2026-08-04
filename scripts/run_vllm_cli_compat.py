#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


IMMUTABLE_INPUT_SCHEMA_VERSION = "immutable-input-attestation/v1"


def _canonicalize_input(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("resolved benchmark input contains a non-finite float")
        return value
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize_input(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_input(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_input(item) for item in value]

    module_name = type(value).__module__
    if module_name.startswith("numpy") and hasattr(value, "tolist"):
        return _canonicalize_input(value.tolist())
    if module_name.startswith("PIL.") and all(
        hasattr(value, attribute) for attribute in ("mode", "size", "tobytes")
    ):
        pixels = value.tobytes()
        return {
            "type": "pil-image",
            "mode": str(value.mode),
            "size": list(value.size),
            "pixels_sha256": hashlib.sha256(pixels).hexdigest(),
        }
    raise TypeError(
        "resolved benchmark input contains an unsupported value: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _sample_request_payload(request: object) -> dict[str, object]:
    return {
        "prompt": _canonicalize_input(getattr(request, "prompt")),
        "prompt_len": int(getattr(request, "prompt_len")),
        "expected_output_len": int(getattr(request, "expected_output_len")),
        "multi_modal_data": _canonicalize_input(
            getattr(request, "multi_modal_data", None)
        ),
        "request_id": _canonicalize_input(getattr(request, "request_id", None)),
    }


def resolved_input_sha256(*, input_kind: str, inputs: object) -> str:
    canonical = {
        "input_kind": input_kind,
        "inputs": _canonicalize_input(inputs),
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def record_immutable_inputs(*, input_kind: str, inputs: object) -> None:
    output_value = os.environ.get("VLLM_HUST_IMMUTABLE_INPUT_ATTESTATION_FILE")
    if not output_value:
        return

    metadata_text = os.environ.get("VLLM_HUST_IMMUTABLE_INPUT_METADATA")
    if metadata_text:
        metadata = json.loads(metadata_text)
        if not isinstance(metadata, dict):
            raise RuntimeError("immutable input metadata must be an object")
        model_id = metadata.get("model_id", "")
        model_revision = metadata.get("model_revision", "")
        data_identity = metadata.get("data_identity")
    else:
        model_id = os.environ.get("VLLM_HUST_MODEL_ID", "")
        model_revision = os.environ.get("VLLM_HUST_MODEL_REVISION", "")
        data_identity_text = os.environ.get("VLLM_HUST_DATA_IDENTITY_JSON", "")
        data_identity = json.loads(data_identity_text) if data_identity_text else None
    if not model_id or not model_revision or not data_identity:
        raise RuntimeError(
            "immutable input capture requires model ID, model revision, and data identity"
        )
    if not isinstance(data_identity, dict) or not data_identity:
        raise RuntimeError("immutable input data identity must be a non-empty object")

    payload = {
        "schema_version": IMMUTABLE_INPUT_SCHEMA_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "data_identity": data_identity,
        "resolved_input_kind": input_kind,
        "resolved_input_sha256": resolved_input_sha256(
            input_kind=input_kind,
            inputs=inputs,
        ),
    }
    output_path = Path(output_value)
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError("benchmark resolved inputs changed within one run")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)


def _record_sample_requests(requests: object, *, input_kind: str) -> None:
    record_immutable_inputs(
        input_kind=input_kind,
        inputs=[_sample_request_payload(request) for request in requests],
    )


def _bound_argument(
    function: object,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    names: tuple[str, ...],
) -> object:
    try:
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        bound = None
    if bound is not None:
        for name in names:
            if name in bound.arguments:
                return bound.arguments[name]
    for name in names:
        if name in kwargs:
            return kwargs[name]
    raise RuntimeError(f"could not resolve input argument {names}")


def _latency_prompt_token_ids(prompts: object) -> list[object]:
    if not isinstance(prompts, (list, tuple)):
        raise TypeError("latency prompts must be a sequence")
    token_ids: list[object] = []
    for prompt in prompts:
        if not isinstance(prompt, Mapping) or "prompt_token_ids" not in prompt:
            raise TypeError("latency LLM.generate input lacks prompt_token_ids")
        token_ids.append(prompt["prompt_token_ids"])
    return token_ids


def install_resolved_input_capture(benchmark_module: Any, benchmark: str) -> None:
    if not os.environ.get("VLLM_HUST_IMMUTABLE_INPUT_ATTESTATION_FILE"):
        return

    if benchmark == "latency":
        from vllm import LLM

        original_generate = LLM.generate

        def generate_with_input_capture(self, *args, **kwargs):
            prompts = _bound_argument(
                original_generate,
                (self, *args),
                kwargs,
                ("prompt_token_ids", "prompts"),
            )
            record_immutable_inputs(
                input_kind="latency-prompt-token-ids",
                inputs=_latency_prompt_token_ids(prompts),
            )
            return original_generate(self, *args, **kwargs)

        LLM.generate = generate_with_input_capture
        return

    if benchmark == "serve":
        original_benchmark = benchmark_module.benchmark

        async def benchmark_with_input_capture(*args, **kwargs):
            requests = _bound_argument(
                original_benchmark,
                args,
                kwargs,
                ("input_requests", "requests"),
            )
            _record_sample_requests(requests, input_kind="serve-sample-requests")
            return await original_benchmark(*args, **kwargs)

        benchmark_module.benchmark = benchmark_with_input_capture
        return

    if benchmark == "throughput":
        original_run_vllm = benchmark_module.run_vllm

        def run_vllm_with_input_capture(*args, **kwargs):
            requests = _bound_argument(
                original_run_vllm,
                args,
                kwargs,
                ("requests", "sample_requests"),
            )
            _record_sample_requests(requests, input_kind="throughput-sample-requests")
            return original_run_vllm(*args, **kwargs)

        benchmark_module.run_vllm = run_vllm_with_input_capture


def restore_huggingface_hub_downloads() -> None:
    try:
        from modelscope.utils.hf_util import unpatch_hub
    except Exception:
        return
    try:
        unpatch_hub()
    except Exception:
        return


def _enum_label(value: object) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def offline_graph_proof(llm: object) -> dict[str, object]:
    vllm_config = llm.llm_engine.vllm_config
    model_config = vllm_config.model_config
    compilation_config = vllm_config.compilation_config
    proof = {
        "schema_version": "vllm-hust-offline-graph-proof/v1",
        "enforce_eager": bool(model_config.enforce_eager),
        "compilation_mode": _enum_label(compilation_config.mode),
        "cudagraph_mode": _enum_label(compilation_config.cudagraph_mode),
    }
    proof["graph_mode_verified"] = (
        not proof["enforce_eager"]
        and proof["compilation_mode"] != "NONE"
        and proof["cudagraph_mode"] != "NONE"
    )
    return proof


def require_offline_graph(proof: dict[str, object]) -> None:
    if proof.get("graph_mode_verified") is True:
        return
    raise RuntimeError(
        "formal offline benchmark resolved to eager/non-graph execution: "
        f"enforce_eager={proof.get('enforce_eager')}, "
        f"compilation_mode={proof.get('compilation_mode')}, "
        f"cudagraph_mode={proof.get('cudagraph_mode')}"
    )


def install_offline_graph_guard() -> None:
    if os.environ.get("VLLM_HUST_REQUIRE_OFFLINE_GRAPH") != "1":
        return

    proof_file = os.environ.get("VLLM_HUST_OFFLINE_GRAPH_PROOF_FILE")
    if not proof_file:
        raise RuntimeError(
            "VLLM_HUST_OFFLINE_GRAPH_PROOF_FILE is required when offline graph "
            "verification is enabled"
        )

    from vllm import LLM

    original_from_engine_args = LLM.from_engine_args

    @classmethod
    def guarded_from_engine_args(_cls, engine_args):
        llm = original_from_engine_args(engine_args)
        proof = offline_graph_proof(llm)
        require_offline_graph(proof)
        output_path = Path(proof_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(proof, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return llm

    LLM.from_engine_args = guarded_from_engine_args


def run_single_benchmark(argv: list[str]) -> int | None:
    if len(argv) < 2 or argv[0] != "bench":
        return None

    benchmark = argv[1]
    if benchmark == "serve":
        from vllm.benchmarks import serve as benchmark_module
    elif benchmark == "latency":
        from vllm.benchmarks import latency as benchmark_module
    elif benchmark == "throughput":
        from vllm.benchmarks import throughput as benchmark_module
    else:
        return None

    add_cli_args = benchmark_module.add_cli_args
    benchmark_main = benchmark_module.main

    try:
        from vllm.utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(prog=f"vllm bench {benchmark}")
    add_cli_args(parser)
    args = parser.parse_args(argv[2:])
    restore_huggingface_hub_downloads()
    if benchmark in {"latency", "throughput"}:
        install_offline_graph_guard()
    install_resolved_input_capture(benchmark_module, benchmark)
    benchmark_main(args)
    return 0


def build_parser():
    from vllm.entrypoints.cli.benchmark import latency, serve, throughput  # noqa: F401
    from vllm.entrypoints.cli.benchmark.main import BenchmarkSubcommand
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup

    try:
        from vllm.utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils.argparse_utils import FlexibleArgumentParser

    cli_env_setup()

    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
    )
    subparsers = parser.add_subparsers(required=True, dest="subparser")

    bench_command = BenchmarkSubcommand()
    bench_command.subparser_init(subparsers).set_defaults(
        dispatch_function=bench_command.cmd,
    )

    return parser, {bench_command.name: bench_command}


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    single_status = run_single_benchmark(argv)
    if single_status is not None:
        return single_status

    parser, commands = build_parser()
    args = parser.parse_args(argv)

    if args.subparser in commands:
        commands[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        args.dispatch_function(args)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
