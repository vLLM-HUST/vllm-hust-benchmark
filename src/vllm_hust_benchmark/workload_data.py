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
    commands.add_parser("delivery-assets")
    source_plan = commands.add_parser("source-plan")
    source_plan.add_argument("source_id")
    classes = commands.add_parser("classes")
    classes.add_argument(
        "--scheme", choices=["research", "delivery-plan"], default="research"
    )
    sources = commands.add_parser("sources")
    sources.add_argument("--class", dest="class_id")
    plan = commands.add_parser("class-plan")
    plan.add_argument("class_id")
    plan.add_argument(
        "--scheme", choices=["research", "delivery-plan"], default="research"
    )
    native = commands.add_parser("native-tasks")
    native.add_argument("--official-archive", type=Path, required=True)
    native.add_argument("--data-root", type=Path, required=True)
    native.add_argument("--task-ids", nargs="+", required=True)
    agentbench = commands.add_parser("agentbench-native-tasks")
    agentbench.add_argument("--official-archive", type=Path, required=True)
    agentbench.add_argument("--game-archives", type=Path, nargs=3, required=True)
    agentbench.add_argument("--data-root", type=Path, required=True)
    agentbench.add_argument("--indices", type=int, nargs="+", required=True)
    longbench = commands.add_parser("prepare-longbench-qasper")
    longbench.add_argument("--data-archive", type=Path, required=True)
    longbench.add_argument("--output", type=Path, required=True)
    bfcl = commands.add_parser("prepare-bfcl-simple-python")
    bfcl.add_argument("--official-archive", type=Path, required=True)
    bfcl.add_argument("--output", type=Path, required=True)
    qrecc = commands.add_parser("prepare-qrecc-test")
    qrecc.add_argument("--data-archive", type=Path, required=True)
    qrecc.add_argument("--output", type=Path, required=True)
    bbh = commands.add_parser("prepare-bbh-boolean")
    bbh.add_argument("--data-archive", type=Path, required=True)
    bbh.add_argument("--output", type=Path, required=True)
    bbh_all = commands.add_parser("prepare-bbh-original")
    bbh_all.add_argument("--data-archive", type=Path, required=True)
    bbh_all.add_argument("--output", type=Path, required=True)
    gpqa = commands.add_parser("prepare-gpqa-original")
    gpqa.add_argument("--data-archive", type=Path, required=True)
    gpqa.add_argument("--output", type=Path, required=True)
    xlsum = commands.add_parser("prepare-xlsum-original")
    xlsum.add_argument("--archive-root", type=Path, required=True)
    xlsum.add_argument("--output", type=Path, required=True)
    beir = commands.add_parser("prepare-beir-original")
    beir.add_argument(
        "--source-id", required=True, choices=["BEIR_SCIFACT", "BEIR_NFCORPUS"]
    )
    beir.add_argument("--data-archive", type=Path, required=True)
    beir.add_argument("--output", type=Path, required=True)
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
        elif args.command == "delivery-assets":
            if not hasattr(provider, "delivery_assets"):
                raise RuntimeError(
                    "install the exact provider containing the 118-source plan"
                )
            value = provider.delivery_assets()
        elif args.command == "source-plan":
            if not hasattr(provider, "delivery_source_plan"):
                raise RuntimeError(
                    "install the exact provider containing the 118-source plan"
                )
            value = provider.delivery_source_plan(args.source_id)
        elif args.command == "classes":
            taxonomy = importlib.import_module("vllm_hust_workload_data.taxonomy")
            if args.scheme == "delivery-plan" and not hasattr(
                taxonomy, "delivery_classes"
            ):
                raise RuntimeError(
                    "install the exact provider containing the 40-class plan; no silent research-view fallback"
                )
            value = (
                taxonomy.delivery_classes()
                if args.scheme == "delivery-plan"
                else taxonomy.research_classes()
            )
        elif args.command == "class-plan":
            if args.scheme == "delivery-plan" and not hasattr(
                provider, "delivery_class_plan"
            ):
                raise RuntimeError(
                    "install the exact provider containing the 40-class plan"
                )
            value = (
                provider.delivery_class_plan(args.class_id)
                if args.scheme == "delivery-plan"
                else provider.class_plan(args.class_id)
            )
        elif args.command == "native-tasks":
            value = provider.tau2_task_references(
                official_archive=args.official_archive,
                data_root=args.data_root,
                task_ids=args.task_ids,
            )
        elif args.command == "agentbench-native-tasks":
            if not hasattr(provider, "agentbench_task_references"):
                raise RuntimeError(
                    "Install the pinned workload-data component from the AgentBench PR; an older 0.1.0 wheel lacks this native source interface"
                )
            value = provider.agentbench_task_references(
                official_archive=args.official_archive,
                game_archives=args.game_archives,
                data_root=args.data_root,
                indices=args.indices,
            )
        elif args.command == "prepare-longbench-qasper":
            if not hasattr(provider, "prepare_longbench_qasper"):
                raise RuntimeError(
                    "Install the pinned LongBench data component; an older wheel lacks this preparation interface"
                )
            value = provider.prepare_longbench_qasper(
                data_archive=args.data_archive, output=args.output
            )
        elif args.command == "prepare-bfcl-simple-python":
            if not hasattr(provider, "prepare_bfcl_simple_python"):
                raise RuntimeError(
                    "Install the exact BFCL data component; old wheels lack this preparation API"
                )
            value = provider.prepare_bfcl_simple_python(
                official_archive=args.official_archive, output=args.output
            )
        elif args.command == "prepare-bbh-boolean":
            if not hasattr(provider, "prepare_bbh_boolean"):
                raise RuntimeError("Install the pinned BBH-capable data component")
            value = provider.prepare_bbh_boolean(
                data_archive=args.data_archive, output=args.output
            )
        elif args.command == "prepare-beir-original":
            from vllm_hust_workload_data.beir import prepare_original_archive

            value = prepare_original_archive(
                source_id=args.source_id,
                data_archive=args.data_archive,
                output=args.output,
            )
        elif args.command == "prepare-xlsum-original":
            from vllm_hust_workload_data.xlsum import prepare_original_archives

            value = prepare_original_archives(
                archive_root=args.archive_root, output=args.output
            )
        elif args.command == "prepare-gpqa-original":
            from vllm_hust_workload_data.gpqa import prepare_original_archive

            value = prepare_original_archive(
                data_archive=args.data_archive, output=args.output
            )
        elif args.command == "prepare-bbh-original":
            if not hasattr(provider, "prepare_bbh_original_tasks"):
                raise RuntimeError(
                    "Install the pinned BBH all-task data component; an older Boolean-only package is insufficient"
                )
            value = provider.prepare_bbh_original_tasks(
                data_archive=args.data_archive, output=args.output
            )
        elif args.command == "prepare-qrecc-test":
            if not hasattr(provider, "prepare_qrecc_test"):
                raise RuntimeError(
                    "Install the pinned text-expansion data component; older wheels lack QReCC preparation"
                )
            value = provider.prepare_qrecc_test(
                data_archive=args.data_archive, output=args.output
            )
        else:
            value = provider.list_sources(class_id=args.class_id)
    except (RuntimeError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(value, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
