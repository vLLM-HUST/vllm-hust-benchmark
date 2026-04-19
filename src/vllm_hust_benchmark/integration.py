from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from vllm_hust_benchmark.models import render_parameter_flags

FLAG_PATTERN = re.compile(r"^\s+--([a-z0-9][a-z0-9-_]*)\b", re.MULTILINE)
DEFAULT_LOCAL_SERVER_HOST = "127.0.0.1"
DEFAULT_LOCAL_SERVER_PORT = 8000
DEFAULT_READY_TIMEOUT_SECONDS = 180.0
FALLBACK_VLLM_BENCH_SERVE_FLAGS = frozenset(
    {
        "backend",
        "base_url",
        "dataset_name",
        "dataset_path",
        "endpoint",
        "hf_split",
        "host",
        "input_len",
        "max_concurrency",
        "num_prompts",
        "num_warmups",
        "output_len",
        "port",
        "random_batch_size",
        "request_rate",
    }
)


def _default_workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RepoLayout:
    workspace_root: Path
    benchmark_repo: Path
    vllm_hust_repo: Path
    website_repo: Path


def resolve_repo_layout() -> RepoLayout:
    workspace_root = Path(
        os.environ.get("VLLM_HUST_WORKSPACE_ROOT") or _default_workspace_root()
    ).resolve()
    benchmark_repo = Path(
        os.environ.get("VLLM_HUST_BENCHMARK_REPO") or workspace_root / "vllm-hust-benchmark"
    ).resolve()
    vllm_hust_repo = Path(
        os.environ.get("VLLM_HUST_REPO") or workspace_root / "vllm-hust"
    ).resolve()
    website_repo = Path(
        os.environ.get("VLLM_HUST_WEBSITE_REPO") or workspace_root / "vllm-hust-website"
    ).resolve()
    return RepoLayout(
        workspace_root=workspace_root,
        benchmark_repo=benchmark_repo,
        vllm_hust_repo=vllm_hust_repo,
        website_repo=website_repo,
    )


def validate_repo_layout(layout: RepoLayout) -> None:
    if not layout.vllm_hust_repo.joinpath("pyproject.toml").is_file():
        raise ValueError(
            f"vllm-hust repository not found or invalid: {layout.vllm_hust_repo}"
        )
    if not layout.vllm_hust_repo.joinpath("benchmarks").is_dir():
        raise ValueError(
            f"vllm-hust benchmarks directory not found: {layout.vllm_hust_repo / 'benchmarks'}"
        )
    if not layout.website_repo.joinpath("scripts", "aggregate_results.py").is_file():
        raise ValueError(
            "vllm-hust-website aggregation script not found: "
            f"{layout.website_repo / 'scripts' / 'aggregate_results.py'}"
        )
    tests_dir = layout.vllm_hust_repo / ".buildkite" / "performance-benchmarks" / "tests"
    if not tests_dir.is_dir():
        raise ValueError(f"vllm-hust performance benchmark tests not found: {tests_dir}")
    suite_script = (
        layout.vllm_hust_repo
        / ".buildkite"
        / "performance-benchmarks"
        / "scripts"
        / "run-performance-benchmarks.sh"
    )
    if not suite_script.is_file():
        raise ValueError(f"vllm-hust performance benchmark suite not found: {suite_script}")


def build_vllm_command(command_args: list[str]) -> list[str]:
    vllm_hust_executable = shutil.which("vllm-hust")
    if vllm_hust_executable:
        return [vllm_hust_executable, *command_args]

    vllm_executable = shutil.which("vllm")
    if vllm_executable:
        return [vllm_executable, *command_args]
    return [sys.executable, "-m", "vllm", *command_args]


@lru_cache(maxsize=8)
def discover_vllm_flags(*command_parts: str) -> frozenset[str]:
    help_command = build_vllm_command([*command_parts, "--help=all"])
    completed = subprocess.run(
        help_command,
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = f"{completed.stdout}\n{completed.stderr}"
    flags = {
        match.group(1).replace("-", "_")
        for match in FLAG_PATTERN.finditer(help_text)
    }
    if not flags:
        if command_parts == ("bench", "serve"):
            return FALLBACK_VLLM_BENCH_SERVE_FLAGS
        raise RuntimeError(
            f"Unable to discover flags for {' '.join(command_parts)}. "
            f"Command exited with code {completed.returncode}."
        )
    return frozenset(flags)


def split_vllm_serve_scenario_parameters(
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bench_flags = discover_vllm_flags("bench", "serve")
    bench_parameters: dict[str, Any] = {}
    serve_parameters: dict[str, Any] = {}
    for key, value in parameters.items():
        if key in bench_flags:
            bench_parameters[key] = value
        else:
            serve_parameters[key] = value
    return bench_parameters, serve_parameters


def build_vllm_bench_command(
    bench_args: list[str],
) -> list[str]:
    return build_vllm_command(["bench", *bench_args])


def build_vllm_serve_command(
    model: str,
    serve_args: list[str],
) -> list[str]:
    return build_vllm_command(["serve", model, *serve_args])


def build_benchmark_script_command(
    layout: RepoLayout,
    script_name: str,
    script_args: list[str],
) -> list[str]:
    script_path = layout.vllm_hust_repo / "benchmarks" / script_name
    if not script_path.is_file():
        raise ValueError(f"benchmark script not found: {script_path}")
    return [sys.executable, str(script_path), *script_args]


def build_performance_suite_command(layout: RepoLayout) -> list[str]:
    suite_script = (
        layout.vllm_hust_repo
        / ".buildkite"
        / "performance-benchmarks"
        / "scripts"
        / "run-performance-benchmarks.sh"
    )
    return ["bash", str(suite_script)]


def _format_env_prefix(env: Mapping[str, object] | None) -> str:
    if not env:
        return ""
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items()) + " "


def run_external_command(
    command: list[str],
    *,
    cwd: Path,
    execute: bool,
    env: Mapping[str, str] | None = None,
) -> int:
    if not execute:
        print(f"{_format_env_prefix(env)}{shlex.join(command)}")
        return 0
    effective_env = os.environ.copy()
    if env:
        effective_env.update(env)
    completed = subprocess.run(command, cwd=cwd, check=False, env=effective_env)
    return completed.returncode


def _resolve_local_base_url(bench_parameters: Mapping[str, Any]) -> str:
    base_url = bench_parameters.get("base_url")
    if isinstance(base_url, str) and base_url.strip():
        return base_url.rstrip("/")
    host = str(bench_parameters.get("host") or DEFAULT_LOCAL_SERVER_HOST)
    port = int(bench_parameters.get("port") or DEFAULT_LOCAL_SERVER_PORT)
    return f"http://{host}:{port}"


def _wait_for_local_server_ready(base_url: str, timeout_seconds: float) -> None:
    ready_urls = (f"{base_url}/health", f"{base_url}/v1/models")
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        for ready_url in ready_urls:
            try:
                with urllib.request.urlopen(ready_url, timeout=5) as response:
                    if response.status < 500:
                        return
            except (OSError, urllib.error.URLError) as error:
                last_error = error
        time.sleep(1)
    raise RuntimeError(
        f"Timed out waiting for local vllm serve at {base_url}"
        + (f": {last_error}" if last_error else "")
    )


def run_local_serve_benchmark(
    *,
    layout: RepoLayout,
    model: str,
    bench_parameters: Mapping[str, Any],
    serve_parameters: Mapping[str, Any],
    env: Mapping[str, str] | None = None,
) -> int:
    serve_command = build_vllm_serve_command(
        model,
        render_parameter_flags(dict(serve_parameters)),
    )
    bench_command = build_vllm_bench_command(
        ["serve", "--model", model, *render_parameter_flags(dict(bench_parameters))]
    )

    effective_env = os.environ.copy()
    if env:
        effective_env.update(env)

    server_process = subprocess.Popen(
        serve_command,
        cwd=layout.vllm_hust_repo,
        env=effective_env,
    )
    try:
        _wait_for_local_server_ready(
            _resolve_local_base_url(bench_parameters),
            timeout_seconds=DEFAULT_READY_TIMEOUT_SECONDS,
        )
        completed = subprocess.run(
            bench_command,
            cwd=layout.vllm_hust_repo,
            check=False,
            env=effective_env,
        )
        return completed.returncode
    finally:
        if server_process.poll() is None:
            server_process.terminate()
            try:
                server_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait(timeout=20)


def aggregate_to_website(
    *,
    layout: RepoLayout,
    source_dir: Path,
    output_dir: Path | None = None,
    execute: bool,
) -> int:
    destination = output_dir or layout.website_repo / "data"
    command = [
        sys.executable,
        str(layout.website_repo / "scripts" / "aggregate_results.py"),
        "--source-dir",
        str(source_dir),
        "--output-dir",
        str(destination),
    ]
    return run_external_command(command, cwd=layout.website_repo, execute=execute)


def upload_to_huggingface(
    *,
    data_dir: Path,
    repo_id: str,
    token: str | None = None,
    branch: str = "main",
    path_in_repo_prefix: str = "",
    commit_message: str = "chore: update leaderboard data",
    dry_run: bool = False,
) -> int:
    """Upload aggregated leaderboard JSON files to a HuggingFace dataset repo.

    Returns 0 on success, non-zero on failure.
    """
    from vllm_hust_benchmark.hf_publisher import upload_leaderboard_to_hf

    try:
        upload_leaderboard_to_hf(
            data_dir=data_dir,
            repo_id=repo_id,
            token=token,
            branch=branch,
            path_in_repo_prefix=path_in_repo_prefix,
            commit_message=commit_message,
            dry_run=dry_run,
        )
        return 0
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"HF upload failed: {exc}", file=sys.stderr)
        return 1
