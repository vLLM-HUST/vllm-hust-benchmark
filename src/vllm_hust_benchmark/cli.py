from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from vllm_hust_benchmark.leaderboard_export import export_leaderboard_artifacts
from vllm_hust_benchmark.registry import filter_scenarios, get_scenario


def _parse_set_arguments(values: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}. Expected key=value")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-benchmark",
        description="Independent benchmark harness mirroring upstream vLLM benchmark entrypoints.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-scenarios", help="List official mirrored scenarios.")
    list_parser.add_argument("--benchmark-type", choices=["serve", "throughput", "latency"])
    list_parser.add_argument("--tag")

    map_parser = subparsers.add_parser(
        "list-leaderboard-map",
        help="List scenario to leaderboard workload/accountable-scope mapping.",
    )
    map_parser.add_argument("--benchmark-type", choices=["serve", "throughput", "latency"])
    map_parser.add_argument("--tag")

    build_parser = subparsers.add_parser("build-command", help="Build the upstream-equivalent vllm bench command for a scenario.")
    build_parser.add_argument("scenario")
    build_parser.add_argument("--model", required=True)
    build_parser.add_argument("--set", action="append", default=[])

    run_parser = subparsers.add_parser("run", help="Run or print the upstream-equivalent vllm bench command for a scenario.")
    run_parser.add_argument("scenario")
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--set", action="append", default=[])
    run_parser.add_argument("--execute", action="store_true")

    analyze_parser = subparsers.add_parser("analyze-upstream", help="Print the upstream benchmark boundary and mirrored design points.")
    analyze_parser.add_argument("--format", choices=["text"], default="text")

    export_parser = subparsers.add_parser(
        "export-leaderboard-artifact",
        help="Export website-compatible leaderboard artifact and manifest from benchmark results.",
    )
    export_parser.add_argument("scenario")
    export_parser.add_argument("--metrics-file", required=True)
    export_parser.add_argument("--output-dir", required=True)
    export_parser.add_argument("--artifact-name", default="run_leaderboard.json")
    export_parser.add_argument("--run-id", required=True)
    export_parser.add_argument("--engine", required=True)
    export_parser.add_argument("--engine-version", required=True)
    export_parser.add_argument("--model-name", required=True)
    export_parser.add_argument("--model-parameters", default="7B")
    export_parser.add_argument("--model-precision", default="BF16")
    export_parser.add_argument("--hardware-vendor", default="Huawei")
    export_parser.add_argument("--hardware-chip-model", required=True)
    export_parser.add_argument("--chip-count", type=int, default=1)
    export_parser.add_argument("--node-count", type=int, default=1)
    export_parser.add_argument("--submitter", required=True)
    export_parser.add_argument("--baseline-engine", default="vllm")
    export_parser.add_argument("--domestic-chip-class", default="Ascend-class")
    export_parser.add_argument("--representative-model-band", default="7B-13B")
    export_parser.add_argument("--data-source", default="vllm-hust-benchmark")
    export_parser.add_argument("--input-length", type=int)
    export_parser.add_argument("--output-length", type=int)
    export_parser.add_argument("--batch-size", type=int)
    export_parser.add_argument("--concurrent-requests", type=int)
    export_parser.add_argument("--protocol-version", default="N/A")
    export_parser.add_argument("--backend-version", default="N/A")
    export_parser.add_argument("--core-version", default="N/A")

    return parser


def _format_scenarios() -> str:
    lines = ["name\tbenchmark\ttags\ttitle"]
    for scenario in filter_scenarios():
        lines.append(
            f"{scenario.name}\t{scenario.benchmark_type}\t{','.join(scenario.tags)}\t{scenario.title}"
        )
    return "\n".join(lines)


def _format_analysis() -> str:
    return "\n".join(
        [
            "Official vLLM benchmark boundary:",
            "- CLI entrypoints: vllm/entrypoints/cli/benchmark/*",
            "- Runtime modules: vllm/benchmarks/*",
            "- Dataset center: vllm/benchmarks/datasets.py",
            "- Shared serving utilities: vllm/benchmarks/lib/*",
            "- Sweep automation: vllm/benchmarks/sweep/*",
            "",
            "Independent repo design in this repository:",
            "- mirror official scenarios as data",
            "- build exact upstream-equivalent commands",
            "- add future scenarios by extending the registry",
            "- keep custom scenario growth outside upstream core paths",
        ]
    )


def _format_leaderboard_map(benchmark_type: str | None, tag: str | None) -> str:
    lines = [
        "name\tbenchmark\tworkload\tbusiness_scenario\tdefault_config_type",
    ]
    for scenario in filter_scenarios(benchmark_type=benchmark_type, tag=tag):
        mapping = scenario.leaderboard
        lines.append(
            "\t".join(
                [
                    scenario.name,
                    scenario.benchmark_type,
                    str(mapping.get("workload_name") or scenario.name),
                    str(mapping.get("representative_business_scenario") or "general-serving"),
                    str(mapping.get("default_config_type") or "single_gpu"),
                ]
            )
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-scenarios":
        scenarios = filter_scenarios(
            benchmark_type=args.benchmark_type,
            tag=args.tag,
        )
        print("name\tbenchmark\ttags\ttitle")
        for scenario in scenarios:
            print(
                f"{scenario.name}\t{scenario.benchmark_type}\t{','.join(scenario.tags)}\t{scenario.title}"
            )
        return 0

    if args.command == "analyze-upstream":
        print(_format_analysis())
        return 0

    if args.command == "list-leaderboard-map":
        print(_format_leaderboard_map(args.benchmark_type, args.tag))
        return 0

    if args.command == "export-leaderboard-artifact":
        scenario = get_scenario(args.scenario)
        try:
            artifact_path, manifest_path = export_leaderboard_artifacts(
                scenario=scenario,
                metrics_file=Path(args.metrics_file),
                output_dir=Path(args.output_dir),
                artifact_name=args.artifact_name,
                run_id=args.run_id,
                engine=args.engine,
                engine_version=args.engine_version,
                model_name=args.model_name,
                model_parameters=args.model_parameters,
                model_precision=args.model_precision,
                hardware_vendor=args.hardware_vendor,
                hardware_chip_model=args.hardware_chip_model,
                chip_count=args.chip_count,
                node_count=args.node_count,
                submitter=args.submitter,
                baseline_engine=args.baseline_engine,
                domestic_chip_class=args.domestic_chip_class,
                representative_model_band=args.representative_model_band,
                data_source=args.data_source,
                input_length=args.input_length,
                output_length=args.output_length,
                batch_size=args.batch_size,
                concurrent_requests=args.concurrent_requests,
                protocol_version=args.protocol_version,
                backend_version=args.backend_version,
                core_version=args.core_version,
            )
        except (OSError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        print(f"artifact: {artifact_path}")
        print(f"manifest: {manifest_path}")
        return 0

    try:
        overrides = _parse_set_arguments(args.set)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    scenario = get_scenario(args.scenario)
    command = scenario.render_command(model=args.model, overrides=overrides)

    if args.command == "build-command":
        print(shlex.join(command))
        return 0

    if args.command == "run":
        if not args.execute:
            print(shlex.join(command))
            return 0
        completed = subprocess.run(command, check=False)
        return completed.returncode

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())