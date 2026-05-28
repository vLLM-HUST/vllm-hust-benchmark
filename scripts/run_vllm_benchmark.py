#!/usr/bin/env python3
from __future__ import annotations

import sys


def build_parser():
    from vllm.entrypoints.cli.benchmark import latency, serve, throughput  # noqa: F401
    from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
    from vllm.entrypoints.utils import cli_env_setup
    from vllm.utils import FlexibleArgumentParser

    cli_env_setup()

    parser = FlexibleArgumentParser(prog="vllm bench")
    subparsers = parser.add_subparsers(required=True, dest="bench_type")

    for command_cls in sorted(BenchmarkSubcommandBase.__subclasses__(),
                              key=lambda cls: cls.name):
        command_parser = subparsers.add_parser(
            command_cls.name,
            help=command_cls.help,
            description=command_cls.help,
            usage=f"vllm bench {command_cls.name} [options]",
        )
        command_parser.set_defaults(dispatch_function=command_cls.cmd)
        command_cls.add_cli_args(command_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.dispatch_function(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))