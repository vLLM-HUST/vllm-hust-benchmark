#!/usr/bin/env python3
from __future__ import annotations

import json
import base64
import dataclasses
import hashlib
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ATTESTATION_SCHEMA = "immutable-input-attestation/v1"


def _canonical_value(value: object) -> object:
    """Return a JSON value whose encoding is stable across Python processes."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value:
            return {"$float": "nan"}
        if value == float("inf"):
            return {"$float": "+inf"}
        if value == float("-inf"):
            return {"$float": "-inf"}
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "$dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": _canonical_value(dataclasses.asdict(value)),
        }
    if isinstance(value, Mapping):
        entries = [
            [_canonical_value(key), _canonical_value(item)]
            for key, item in value.items()
        ]
        entries.sort(key=lambda item: canonical_json_bytes(item[0]))
        return {"$mapping": entries}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        items.sort(key=canonical_json_bytes)
        return {"$set": items}

    # numpy arrays/scalars: include the logical values, dtype and shape.  This
    # avoids platform-dependent repr output and object-array pointer bytes.
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy":
        if hasattr(value, "shape") and hasattr(value, "tolist"):
            return {
                "$numpy": {
                    "dtype": str(getattr(value, "dtype", "")),
                    "shape": list(getattr(value, "shape", ())),
                    "values": _canonical_value(value.tolist()),
                }
            }
        if hasattr(value, "item"):
            return _canonical_value(value.item())

    # PIL images are identified by decoded pixels, not container metadata or
    # repr (which includes a memory address on some Pillow versions).
    if module == "PIL" and hasattr(value, "tobytes"):
        pixels = value.tobytes()
        return {
            "$pil_image": {
                "mode": str(getattr(value, "mode", "")),
                "size": list(getattr(value, "size", ())),
                "pixels_sha256": hashlib.sha256(pixels).hexdigest(),
            }
        }
    if hasattr(value, "__dict__"):
        return {
            "$object": f"{type(value).__module__}.{type(value).__qualname__}",
            "attributes": _canonical_value(vars(value)),
        }
    raise TypeError(f"unsupported immutable input value: {type(value)!r}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _latency_prompt_token_ids(prompts: object) -> object:
    if not isinstance(prompts, (list, tuple)):
        raise TypeError("latency prompts must be a sequence")
    token_ids = []
    for prompt in prompts:
        if not isinstance(prompt, Mapping) or "prompt_token_ids" not in prompt:
            raise TypeError("latency LLM.generate input lacks prompt_token_ids")
        token_ids.append(prompt["prompt_token_ids"])
    return token_ids


class ImmutableInputRecorder:
    def __init__(self, output_file: Path, metadata: Mapping[str, object]) -> None:
        self.output_file = output_file
        self.metadata = dict(metadata)
        self._kind: str | None = None
        self._sha256: str | None = None

    @classmethod
    def from_environment(cls) -> "ImmutableInputRecorder | None":
        output = os.environ.get("VLLM_HUST_IMMUTABLE_INPUT_ATTESTATION_FILE")
        if not output:
            return None
        metadata_text = os.environ.get("VLLM_HUST_IMMUTABLE_INPUT_METADATA")
        if not metadata_text:
            raise RuntimeError("VLLM_HUST_IMMUTABLE_INPUT_METADATA is required")
        metadata = json.loads(metadata_text)
        required = {"model_id", "model_revision", "data_identity"}
        missing = sorted(required - metadata.keys())
        if missing:
            raise RuntimeError(f"immutable input metadata missing: {missing}")
        return cls(Path(output), metadata)

    def record(self, kind: str, value: object) -> None:
        digest = canonical_sha256(value)
        if self._sha256 is not None and (kind, digest) != (self._kind, self._sha256):
            raise RuntimeError(
                "immutable benchmark input drifted within one process: "
                f"{self._kind}:{self._sha256} != {kind}:{digest}"
            )
        self._kind = kind
        self._sha256 = digest

    def write(self) -> None:
        if self._kind is None or self._sha256 is None:
            raise RuntimeError("benchmark did not expose a real immutable input")
        payload = {
            "schema_version": ATTESTATION_SCHEMA,
            **self.metadata,
            "resolved_input_kind": self._kind,
            "resolved_input_sha256": self._sha256,
        }
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_file.with_suffix(self.output_file.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(self.output_file)


def _bound_argument(
    function: object,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    names: tuple[str, ...],
) -> object:
    import inspect

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
    if args:
        return args[0]
    raise RuntimeError(f"could not resolve input argument {names}")


def install_immutable_input_capture(
    benchmark: str, module: object
) -> tuple[ImmutableInputRecorder | None, list[object]]:
    recorder = ImmutableInputRecorder.from_environment()
    serve_requests: list[object] = []
    if recorder is None:
        return None, serve_requests

    if benchmark == "latency":
        from vllm import LLM

        original_generate = LLM.generate

        def captured_generate(self, *args, **kwargs):
            prompts = _bound_argument(
                original_generate,
                (self, *args),
                kwargs,
                ("prompt_token_ids", "prompts"),
            )
            recorder.record(
                "latency-prompt-token-ids", _latency_prompt_token_ids(prompts)
            )
            return original_generate(self, *args, **kwargs)

        LLM.generate = captured_generate
    elif benchmark == "throughput":
        original_run_vllm = getattr(module, "run_vllm")

        def captured_run_vllm(*args, **kwargs):
            requests = _bound_argument(
                original_run_vllm, args, kwargs, ("requests", "sample_requests")
            )
            recorder.record("throughput-sample-requests", requests)
            return original_run_vllm(*args, **kwargs)

        setattr(module, "run_vllm", captured_run_vllm)
    elif benchmark == "serve":
        original_benchmark = getattr(module, "benchmark")

        async def captured_benchmark(*args, **kwargs):
            requests = _bound_argument(
                original_benchmark, args, kwargs, ("input_requests", "requests")
            )
            serve_requests.extend(requests)
            recorder.record("serve-sample-requests", requests)
            return await original_benchmark(*args, **kwargs)

        setattr(module, "benchmark", captured_benchmark)
    return recorder, serve_requests


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
        import vllm.benchmarks.serve as benchmark_module
    elif benchmark == "latency":
        import vllm.benchmarks.latency as benchmark_module
    elif benchmark == "throughput":
        import vllm.benchmarks.throughput as benchmark_module
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
    recorder, serve_requests = install_immutable_input_capture(
        benchmark, benchmark_module
    )
    benchmark_main(args)
    if recorder is not None:
        recorder.write()
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
