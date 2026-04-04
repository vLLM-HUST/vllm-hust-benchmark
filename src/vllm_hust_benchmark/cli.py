from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from vllm_hust_benchmark.integration import (
    aggregate_to_website,
    build_benchmark_script_command,
    build_performance_suite_command,
    build_vllm_bench_command,
    build_vllm_serve_command,
    resolve_repo_layout,
    run_external_command,
    run_local_serve_benchmark,
    split_vllm_serve_scenario_parameters,
    validate_repo_layout,
)
from vllm_hust_benchmark.leaderboard_export import export_leaderboard_artifacts
from vllm_hust_benchmark.models import render_parameter_flags
from vllm_hust_benchmark.registry import filter_scenarios, get_scenario
from vllm_hust_benchmark.upstream_tests import (
    build_inspection_commands,
    get_upstream_test,
    load_upstream_tests,
)


def _parse_override_value(raw_value: str) -> object:
    value = raw_value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_set_arguments(values: list[str] | None) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}. Expected key=value")
        key, value = item.split("=", 1)
        normalized_key = key.strip().replace("-", "_")
        parsed[normalized_key] = _parse_override_value(value)
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-benchmark",
        description="Independent benchmark harness mirroring upstream vLLM benchmark entrypoints.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    repos_parser = subparsers.add_parser(
        "show-repos",
        help="Show the resolved sibling repository layout used by the wrapper.",
    )
    repos_parser.add_argument("--validate", action="store_true")

    tests_parser = subparsers.add_parser(
        "list-tests",
        help="List official upstream benchmark tests from the sibling vllm-hust repo.",
    )
    tests_parser.add_argument("--benchmark-type", choices=["serve", "throughput", "latency"])

    show_test_parser = subparsers.add_parser(
        "show-test",
        help="Show the resolved upstream benchmark test definition and wrapped commands.",
    )
    show_test_parser.add_argument("test_name")
    show_test_parser.add_argument("--qps")
    show_test_parser.add_argument("--max-concurrency", type=int)
    show_test_parser.add_argument("--result-json")

    run_test_parser = subparsers.add_parser(
        "run-test",
        help="Run one upstream benchmark test by delegating to vllm-hust's benchmark suite.",
    )
    run_test_parser.add_argument("test_name")
    run_test_parser.add_argument("--execute", action="store_true")
    run_test_parser.add_argument("--env", action="append", default=[])

    suite_parser = subparsers.add_parser(
        "run-suite",
        help="Run the upstream vllm-hust performance benchmark suite from this wrapper repo.",
    )
    suite_parser.add_argument("--execute", action="store_true")
    suite_parser.add_argument("--env", action="append", default=[])

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

    bench_parser = subparsers.add_parser(
        "bench",
        help="Run or print a vllm-hust 'vllm bench ...' command from the sibling vllm-hust repo.",
    )
    bench_parser.add_argument("bench_args", nargs=argparse.REMAINDER)
    bench_parser.add_argument("--execute", action="store_true")

    script_parser = subparsers.add_parser(
        "run-script",
        help="Run or print a concrete script from vllm-hust/benchmarks/.",
    )
    script_parser.add_argument("script_name")
    script_parser.add_argument("script_args", nargs=argparse.REMAINDER)
    script_parser.add_argument("--execute", action="store_true")

    analyze_parser = subparsers.add_parser("analyze-upstream", help="Print the upstream benchmark boundary and mirrored design points.")
    analyze_parser.add_argument("--format", choices=["text"], default="text")

    export_parser = subparsers.add_parser(
        "export-leaderboard-artifact",
        help="Export website-compatible leaderboard artifact and manifest from benchmark results.",
    )
    export_parser.add_argument("scenario")
    export_parser.add_argument("--metrics-file")
    export_parser.add_argument("--benchmark-result-file")
    export_parser.add_argument("--constraints-file")
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
    export_parser.add_argument("--peak-mem-mb", type=float)
    export_parser.add_argument("--publish-website", action="store_true")
    export_parser.add_argument("--website-output-dir")
    export_parser.add_argument("--execute", action="store_true")

    publish_parser = subparsers.add_parser(
        "publish-website",
        help="Aggregate exported leaderboard artifacts into vllm-hust-website data outputs.",
    )
    publish_parser.add_argument("--source-dir", required=True)
    publish_parser.add_argument("--output-dir")
    publish_parser.add_argument("--execute", action="store_true")

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


def _format_upstream_tests(layout, benchmark_type: str | None) -> str:
    lines = ["name\tbenchmark\tsource"]
    for test in load_upstream_tests(layout, benchmark_type=benchmark_type):
        lines.append(
            f"{test.name}\t{test.benchmark_type}\t{test.source_file.name}"
        )
    return "\n".join(lines)


def _format_test_details(test, commands: dict[str, list[str]]) -> str:
    lines = [
        f"name: {test.name}",
        f"benchmark_type: {test.benchmark_type}",
        f"source_file: {test.source_file}",
    ]
    if test.server_environment_variables:
        lines.append(f"server_environment_variables: {test.server_environment_variables}")
    if test.server_parameters:
        lines.append(f"server_parameters: {test.server_parameters}")
    if test.client_parameters:
        lines.append(f"client_parameters: {test.client_parameters}")
    if test.parameters:
        lines.append(f"parameters: {test.parameters}")
    if test.qps_list:
        lines.append(f"qps_list: {list(test.qps_list)}")
    if test.max_concurrency_list:
        lines.append(f"max_concurrency_list: {list(test.max_concurrency_list)}")
    for name, command in commands.items():
        lines.append(f"{name}_command: {shlex.join(command)}")
    return "\n".join(lines)


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

    if args.command == "show-repos":
        layout = resolve_repo_layout()
        if args.validate:
            try:
                validate_repo_layout(layout)
            except ValueError as error:
                print(str(error), file=sys.stderr)
                return 2
        print(f"workspace_root: {layout.workspace_root}")
        print(f"benchmark_repo: {layout.benchmark_repo}")
        print(f"vllm_hust_repo: {layout.vllm_hust_repo}")
        print(f"website_repo: {layout.website_repo}")
        return 0

    if args.command == "list-tests":
        layout = resolve_repo_layout()
        try:
            validate_repo_layout(layout)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        print(_format_upstream_tests(layout, args.benchmark_type))
        return 0

    if args.command == "show-test":
        layout = resolve_repo_layout()
        try:
            validate_repo_layout(layout)
            test = get_upstream_test(layout, args.test_name)
        except (KeyError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        result_json = Path(args.result_json).resolve() if args.result_json else None
        commands = build_inspection_commands(
            test,
            result_json=result_json,
            qps=args.qps,
            max_concurrency=args.max_concurrency,
        )
        print(_format_test_details(test, commands))
        return 0

    if args.command == "run-test":
        layout = resolve_repo_layout()
        try:
            validate_repo_layout(layout)
            test = get_upstream_test(layout, args.test_name)
            extra_env = _parse_set_arguments(args.env)
        except (KeyError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        env = {"TEST_SELECTOR": f"^{test.name}$", **extra_env}
        command = build_performance_suite_command(layout)
        return run_external_command(
            command,
            cwd=layout.vllm_hust_repo,
            execute=args.execute,
            env=env,
        )

    if args.command == "run-suite":
        layout = resolve_repo_layout()
        try:
            validate_repo_layout(layout)
            env = _parse_set_arguments(args.env)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        command = build_performance_suite_command(layout)
        return run_external_command(
            command,
            cwd=layout.vllm_hust_repo,
            execute=args.execute,
            env=env,
        )

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
        metrics_file = Path(args.metrics_file).resolve() if args.metrics_file else None
        benchmark_result_file = (
            Path(args.benchmark_result_file).resolve()
            if args.benchmark_result_file
            else None
        )
        constraints_file = (
            Path(args.constraints_file).resolve() if args.constraints_file else None
        )
        try:
            artifact_path, manifest_path = export_leaderboard_artifacts(
                scenario=scenario,
                metrics_file=metrics_file,
                benchmark_result_file=benchmark_result_file,
                constraints_file=constraints_file,
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
                peak_mem_mb=args.peak_mem_mb,
            )
        except (OSError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 2
        print(f"artifact: {artifact_path}")
        print(f"manifest: {manifest_path}")
        if args.publish_website:
            layout = resolve_repo_layout()
            try:
                validate_repo_layout(layout)
                website_output_dir = (
                    Path(args.website_output_dir).resolve()
                    if args.website_output_dir
                    else None
                )
                aggregate_exit_code = aggregate_to_website(
                    layout=layout,
                    source_dir=Path(args.output_dir).resolve(),
                    output_dir=website_output_dir,
                    execute=args.execute,
                )
            except ValueError as error:
                print(str(error), file=sys.stderr)
                return 2
            return aggregate_exit_code
        return 0

    if args.command == "publish-website":
        layout = resolve_repo_layout()
        try:
            validate_repo_layout(layout)
            output_dir = Path(args.output_dir).resolve() if args.output_dir else None
            return aggregate_to_website(
                layout=layout,
                source_dir=Path(args.source_dir).resolve(),
                output_dir=output_dir,
                execute=args.execute,
            )
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2

    if args.command == "bench":
        layout = resolve_repo_layout()
        try:
            validate_repo_layout(layout)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        bench_args = list(args.bench_args)
        if bench_args and bench_args[0] == "--":
            bench_args = bench_args[1:]
        command = build_vllm_bench_command(bench_args)
        return run_external_command(command, cwd=layout.vllm_hust_repo, execute=args.execute)

    if args.command == "run-script":
        layout = resolve_repo_layout()
        try:
            validate_repo_layout(layout)
            script_args = list(args.script_args)
            if script_args and script_args[0] == "--":
                script_args = script_args[1:]
            command = build_benchmark_script_command(layout, args.script_name, script_args)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        return run_external_command(command, cwd=layout.vllm_hust_repo, execute=args.execute)

    try:
        overrides = _parse_set_arguments(args.set)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2

    scenario = get_scenario(args.scenario)
    merged_parameters = scenario.merge_parameters(overrides)

    if scenario.benchmark_type == "serve":
        bench_parameters, serve_parameters = split_vllm_serve_scenario_parameters(
            merged_parameters
        )
        command = [
            "vllm",
            "bench",
            "serve",
            "--model",
            args.model,
            *render_parameter_flags(bench_parameters),
        ]

        if serve_parameters:
            server_command = build_vllm_serve_command(
                args.model, render_parameter_flags(serve_parameters)
            )
            client_command = build_vllm_bench_command(command[2:])
            if args.command == "build-command" or not args.execute:
                print(f"server_command: {shlex.join(server_command)}")
                print(f"client_command: {shlex.join(client_command)}")
                return 0

            layout = resolve_repo_layout()
            try:
                validate_repo_layout(layout)
            except ValueError as error:
                print(str(error), file=sys.stderr)
                return 2
            return run_local_serve_benchmark(
                layout=layout,
                model=args.model,
                bench_parameters=bench_parameters,
                serve_parameters=serve_parameters,
            )
    else:
        command = scenario.render_command(model=args.model, overrides=overrides)

    if args.command == "build-command":
        print(shlex.join(command))
        return 0

    if args.command == "run":
        if not args.execute:
            print(shlex.join(command))
            return 0
        layout = resolve_repo_layout()
        try:
            validate_repo_layout(layout)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 2
        return run_external_command(
            build_vllm_bench_command(command[2:]),
            cwd=layout.vllm_hust_repo,
            execute=True,
        )

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())