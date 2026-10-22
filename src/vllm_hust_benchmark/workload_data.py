"""Optional workload provider entrypoint; no data/model download on import."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import runpy
import sys


def load_provider():
    try:
        provider = importlib.import_module("vllm_hust_workload_data.provider")
    except ModuleNotFoundError as error:
        if error.name not in {
            "vllm_hust_workload_data",
            "vllm_hust_workload_data.provider",
        }:
            raise
        raise RuntimeError(
            "Install this repository's vllm-hust-workload-data component (version 0.1.0); "
            "an old research editable checkout is not a substitute."
        ) from None
    identity = provider.provider_identity()
    if (
        identity["contract"] != "benchmark-workload-provider/v1"
        or identity["version"] != "0.1.0"
    ):
        raise RuntimeError(
            "unsupported workload provider; install the pinned benchmark component"
        )
    return provider


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("identity")
    commands.add_parser("classes")
    sources = commands.add_parser("sources")
    sources.add_argument("--class", dest="class_id")
    plan = commands.add_parser("class-plan")
    plan.add_argument("class_id")
    native = commands.add_parser("native-tasks")
    native.add_argument("--official-archive", type=Path, required=True)
    native.add_argument("--data-root", type=Path, required=True)
    native.add_argument("--task-ids", nargs="+", required=True)
    for name in ("catalog", "pipeline", "replay"):
        passthrough = commands.add_parser(name, add_help=False)
        passthrough.add_argument("arguments", nargs=argparse.REMAINDER)
    try:
        # Delegate existing CLI parsers, including their own --help, unchanged.
        if arguments and arguments[0] in {"catalog", "pipeline", "replay"}:
            load_provider()
            name = arguments[0]
            module = {
                "catalog": "catalog",
                "pipeline": "pipeline",
                "replay": "execution",
            }[name]
            original = sys.argv
            try:
                sys.argv = [f"workload-data {name}", *arguments[1:]]
                runpy.run_module(
                    f"vllm_hust_workload_data.{module}", run_name="__main__"
                )
            finally:
                sys.argv = original
            return 0
        args = parser.parse_args(arguments)
        provider = load_provider()
        if args.command == "identity":
            value = provider.provider_identity()
        elif args.command == "classes":
            value = importlib.import_module(
                "vllm_hust_workload_data.taxonomy"
            ).research_classes()
        elif args.command == "class-plan":
            value = provider.class_plan(args.class_id)
        elif args.command == "native-tasks":
            value = provider.tau2_task_references(
                official_archive=args.official_archive,
                data_root=args.data_root,
                task_ids=args.task_ids,
            )
        else:
            value = provider.list_sources(class_id=args.class_id)
    except (RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(value, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
