from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.integration import RepoLayout


DEFAULT_TEST_FILES: dict[str, str] = {
    "serve": "serving-tests.json",
    "latency": "latency-tests.json",
    "throughput": "throughput-tests.json",
}


@dataclass(frozen=True)
class UpstreamBenchmarkTest:
    name: str
    benchmark_type: str
    source_file: Path
    parameters: dict[str, Any] = field(default_factory=dict)
    server_parameters: dict[str, Any] = field(default_factory=dict)
    server_environment_variables: dict[str, Any] = field(default_factory=dict)
    client_parameters: dict[str, Any] = field(default_factory=dict)
    qps_list: tuple[Any, ...] = field(default_factory=tuple)
    max_concurrency_list: tuple[Any, ...] = field(default_factory=tuple)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(override)
    return merged


def _iter_serving_tests(source_file: Path) -> list[UpstreamBenchmarkTest]:
    payload = _load_json(source_file)
    if isinstance(payload, list):
        defaults: dict[str, Any] = {}
        items = payload
    elif isinstance(payload, dict) and isinstance(payload.get("tests"), list):
        defaults = dict(payload.get("defaults") or {})
        items = payload["tests"]
    else:
        raise ValueError(
            f"Unsupported serving tests format in {source_file}. Expected list or object with 'tests'."
        )

    default_server_envs = dict(defaults.get("server_environment_variables") or {})
    default_server_params = dict(defaults.get("server_parameters") or {})
    default_client_params = dict(defaults.get("client_parameters") or {})
    default_qps_list = tuple(defaults.get("qps_list") or ())
    default_max_concurrency_list = tuple(defaults.get("max_concurrency_list") or ())

    tests: list[UpstreamBenchmarkTest] = []
    for item in items:
        tests.append(
            UpstreamBenchmarkTest(
                name=str(item["test_name"]),
                benchmark_type="serve",
                source_file=source_file,
                server_environment_variables=_merge_dicts(
                    default_server_envs,
                    dict(item.get("server_environment_variables") or {}),
                ),
                server_parameters=_merge_dicts(
                    default_server_params,
                    dict(item.get("server_parameters") or {}),
                ),
                client_parameters=_merge_dicts(
                    default_client_params,
                    dict(item.get("client_parameters") or {}),
                ),
                qps_list=tuple(item.get("qps_list") or default_qps_list),
                max_concurrency_list=tuple(
                    item.get("max_concurrency_list") or default_max_concurrency_list
                ),
            )
        )
    return tests


def _iter_simple_tests(source_file: Path, benchmark_type: str) -> list[UpstreamBenchmarkTest]:
    payload = _load_json(source_file)
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported {benchmark_type} tests format in {source_file}.")

    tests: list[UpstreamBenchmarkTest] = []
    for item in payload:
        tests.append(
            UpstreamBenchmarkTest(
                name=str(item["test_name"]),
                benchmark_type=benchmark_type,
                source_file=source_file,
                parameters=dict(item.get("parameters") or {}),
            )
        )
    return tests


def load_upstream_tests(
    layout: RepoLayout,
    *,
    benchmark_type: str | None = None,
) -> list[UpstreamBenchmarkTest]:
    tests_root = layout.vllm_hust_repo / ".buildkite" / "performance-benchmarks" / "tests"
    selected_types = [benchmark_type] if benchmark_type else list(DEFAULT_TEST_FILES)
    tests: list[UpstreamBenchmarkTest] = []
    for current_type in selected_types:
        file_name = DEFAULT_TEST_FILES.get(current_type)
        if file_name is None:
            raise ValueError(f"Unsupported benchmark type: {current_type}")
        source_file = tests_root / file_name
        if current_type == "serve":
            tests.extend(_iter_serving_tests(source_file))
        else:
            tests.extend(_iter_simple_tests(source_file, current_type))
    return tests


def get_upstream_test(layout: RepoLayout, name: str) -> UpstreamBenchmarkTest:
    for test in load_upstream_tests(layout):
        if test.name == name:
            return test
    raise KeyError(f"Unknown upstream test: {name}")


def _render_flag_args(parameters: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in parameters.items():
        if value is None or value is False:
            continue
        flag = f"--{key.replace('_', '-')}"
        if value == "" or value is True:
            args.append(flag)
            continue
        if isinstance(value, (dict, list)):
            args.extend([flag, json.dumps(value, separators=(",", ":"))])
            continue
        args.extend([flag, str(value)])
    return args


def build_inspection_commands(
    test: UpstreamBenchmarkTest,
    *,
    result_json: Path | None = None,
    qps: str | None = None,
    max_concurrency: int | None = None,
) -> dict[str, list[str]]:
    if test.benchmark_type == "serve":
        server_parameters = dict(test.server_parameters)
        server_model = str(server_parameters.pop("model"))
        commands: dict[str, list[str]] = {
            "server": ["vllm", "serve", server_model, *_render_flag_args(server_parameters)],
        }

        client_parameters = dict(test.client_parameters)
        client_command = ["vllm", "bench", "serve"]
        if result_json is not None:
            client_command.extend(["--save-result", "--result-dir", str(result_json.parent), "--result-filename", result_json.name])
        if qps is not None:
            client_command.extend(["--request-rate", qps])
        elif test.qps_list:
            client_command.extend(["--request-rate", str(test.qps_list[0])])
        if max_concurrency is not None:
            client_command.extend(["--max-concurrency", str(max_concurrency)])
        elif test.max_concurrency_list:
            client_command.extend(["--max-concurrency", str(test.max_concurrency_list[0])])
        client_command.extend(_render_flag_args(client_parameters))
        commands["client"] = client_command
        return commands

    command = ["vllm", "bench", test.benchmark_type]
    if result_json is not None:
        command.extend(["--output-json", str(result_json)])
    command.extend(_render_flag_args(test.parameters))
    return {"command": command}