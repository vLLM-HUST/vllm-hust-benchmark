#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


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

    def record_proof(llm: object) -> None:
        proof = offline_graph_proof(llm)
        require_offline_graph(proof)
        output_path = Path(proof_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(proof, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if hasattr(LLM, "from_engine_args"):
        original_from_engine_args = LLM.from_engine_args

        @classmethod
        def guarded_from_engine_args(_cls, engine_args):
            llm = original_from_engine_args(engine_args)
            record_proof(llm)
            return llm

        LLM.from_engine_args = guarded_from_engine_args
        return

    original_init = LLM.__init__

    def guarded_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        record_proof(self)

    LLM.__init__ = guarded_init


def run_single_benchmark(argv: list[str]) -> int | None:
    if len(argv) < 2 or argv[0] != "bench":
        return None

    benchmark = argv[1]
    if benchmark == "serve":
        from vllm.benchmarks.serve import add_cli_args, main
    elif benchmark == "latency":
        from vllm.benchmarks.latency import add_cli_args, main
    elif benchmark == "throughput":
        from vllm.benchmarks.throughput import add_cli_args, main
    else:
        return None

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
    main(args)
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
