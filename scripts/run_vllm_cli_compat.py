#!/usr/bin/env python3
from __future__ import annotations

import sys


def build_parser():
    from vllm.entrypoints.cli.benchmark import latency, serve, throughput  # noqa: F401
    from vllm.entrypoints.cli.benchmark.main import BenchmarkSubcommand
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils import FlexibleArgumentParser

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