from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
from collections.abc import Sequence
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from vllm_hust_benchmark.models import render_parameter_flags

FLAG_PATTERN = re.compile(r"^\s+--([a-z0-9][a-z0-9-_]*)\b", re.MULTILINE)
DEFAULT_RUNTIME_ENGINE = "vllm-hust"
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
    reference_vllm_repo: Path | None = None


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
    reference_vllm_repo = Path(
        os.environ.get("VLLM_BASELINE_VLLM_REPO")
        or workspace_root / "reference-repos" / "vllm"
    ).resolve()
    website_repo = Path(
        os.environ.get("VLLM_HUST_WEBSITE_REPO") or workspace_root / "vllm-hust-website"
    ).resolve()
    return RepoLayout(
        workspace_root=workspace_root,
        benchmark_repo=benchmark_repo,
        vllm_hust_repo=vllm_hust_repo,
        reference_vllm_repo=reference_vllm_repo,
        website_repo=website_repo,
    )


def resolve_runtime_repo(
    layout: RepoLayout,
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
) -> Path:
    if runtime_engine == "vllm-hust":
        return layout.vllm_hust_repo
    if runtime_engine == "vllm":
        if layout.reference_vllm_repo is not None:
            return layout.reference_vllm_repo
        return (layout.workspace_root / "reference-repos" / "vllm").resolve()
    raise ValueError(f"Unsupported runtime engine: {runtime_engine}")


def validate_runtime_repo(
    layout: RepoLayout,
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
    *,
    require_benchmarks: bool = False,
) -> Path:
    runtime_repo = resolve_runtime_repo(layout, runtime_engine)
    if not runtime_repo.joinpath("pyproject.toml").is_file():
        raise ValueError(
            f"{runtime_engine} runtime repository not found or invalid: {runtime_repo}"
        )
    if require_benchmarks and not runtime_repo.joinpath("benchmarks").is_dir():
        raise ValueError(
            f"{runtime_engine} benchmarks directory not found: {runtime_repo / 'benchmarks'}"
        )
    return runtime_repo


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


def build_vllm_command(
    command_args: list[str],
    *,
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
) -> list[str]:
    executable_names = ["vllm-hust", "vllm"]
    if runtime_engine == "vllm":
        executable_names = ["vllm"]
    elif runtime_engine != "vllm-hust":
        raise ValueError(f"Unsupported runtime engine: {runtime_engine}")

    for executable_name in executable_names:
        executable_path = shutil.which(executable_name)
        if executable_path:
            return [executable_path, *command_args]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", *command_args]


@lru_cache(maxsize=8)
def discover_vllm_flags(
    *command_parts: str,
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
    runtime_repo: str | None = None,
) -> frozenset[str]:
    help_command = build_vllm_command(
        [*command_parts, "--help=all"],
        runtime_engine=runtime_engine,
    )
    layout = resolve_repo_layout()
    cwd = Path(runtime_repo).resolve() if runtime_repo else resolve_runtime_repo(layout, runtime_engine)
    completed = subprocess.run(
        help_command,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
        env=_build_effective_env(cwd, None),
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
    *,
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
    runtime_repo: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if runtime_engine == DEFAULT_RUNTIME_ENGINE and runtime_repo is None:
        bench_flags = discover_vllm_flags("bench", "serve")
    else:
        bench_flags = discover_vllm_flags(
            "bench",
            "serve",
            runtime_engine=runtime_engine,
            runtime_repo=runtime_repo,
        )
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
    *,
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
) -> list[str]:
    return build_vllm_command(["bench", *bench_args], runtime_engine=runtime_engine)


def build_vllm_serve_command(
    model: str,
    serve_args: list[str],
    *,
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
) -> list[str]:
    return build_vllm_command(
        ["serve", model, *serve_args],
        runtime_engine=runtime_engine,
    )


def build_benchmark_script_command(
    layout: RepoLayout,
    script_name: str,
    script_args: list[str],
    *,
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
) -> list[str]:
    runtime_repo = validate_runtime_repo(layout, runtime_engine, require_benchmarks=True)
    script_path = runtime_repo / "benchmarks" / script_name
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


def _build_effective_env(
    cwd: Path,
    env: Mapping[str, object] | None,
) -> dict[str, str]:
    effective_env = {key: str(value) for key, value in os.environ.items()}
    if env:
        effective_env.update({key: str(value) for key, value in env.items()})

    cwd_str = str(cwd)
    existing_pythonpath = effective_env.get("PYTHONPATH", "")
    pythonpath_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
    if cwd_str not in pythonpath_entries:
        pythonpath_entries.insert(0, cwd_str)
    effective_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return effective_env


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
    effective_env = _build_effective_env(cwd, env)
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
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
) -> int:
    runtime_repo = validate_runtime_repo(layout, runtime_engine)
    serve_command = build_vllm_serve_command(
        model,
        render_parameter_flags(dict(serve_parameters)),
        runtime_engine=runtime_engine,
    )
    bench_command = build_vllm_bench_command(
        ["serve", "--model", model, *render_parameter_flags(dict(bench_parameters))],
        runtime_engine=runtime_engine,
    )

    effective_env = _build_effective_env(runtime_repo, env)

    server_process = subprocess.Popen(
        serve_command,
        cwd=runtime_repo,
        env=effective_env,
    )
    try:
        _wait_for_local_server_ready(
            _resolve_local_base_url(bench_parameters),
            timeout_seconds=DEFAULT_READY_TIMEOUT_SECONDS,
        )
        completed = subprocess.run(
            bench_command,
            cwd=runtime_repo,
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


def _resolve_hf_token(token: str | None) -> str | None:
    return token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def sync_submission_to_huggingface(
    *,
    layout: RepoLayout,
    submission_dirs: Path | Sequence[Path],
    aggregate_output_dir: Path,
    repo_id: str,
    token: str | None = None,
    branch: str = "main",
    submissions_prefix: str = "submissions-auto",
    commit_message: str = "chore: sync benchmark submission and leaderboard data",
    dry_run: bool = False,
) -> int:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        from vllm_hust_benchmark.hf_publisher import _create_commit_on_branch
    except ImportError:
        print(
            "huggingface_hub is required for HF submission sync. Install with: "
            "pip install 'vllm-hust-benchmark[publish]'",
            file=sys.stderr,
        )
        return 2

    if isinstance(submission_dirs, Path):
        normalized_submission_dirs = [submission_dirs]
    else:
        normalized_submission_dirs = list(submission_dirs)

    if not normalized_submission_dirs:
        print("at least one submission directory is required", file=sys.stderr)
        return 2

    for submission_dir in normalized_submission_dirs:
        if not submission_dir.is_dir():
            print(f"submission directory not found: {submission_dir}", file=sys.stderr)
            return 2

    resolved_token = _resolve_hf_token(token)
    api = HfApi(token=resolved_token)
    normalized_prefix = submissions_prefix.strip("/")
    repo_prefix = f"{normalized_prefix}/" if normalized_prefix else ""

    with tempfile.TemporaryDirectory(prefix="vllm-hust-hf-sync-") as temp_dir:
        temp_root = Path(temp_dir)
        merged_root = temp_root / "merged_submissions"
        merged_source_dir = merged_root / normalized_prefix if normalized_prefix else merged_root
        merged_source_dir.mkdir(parents=True, exist_ok=True)

        try:
            repo_files = api.list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
                revision=branch,
            )
        except Exception:
            repo_files = []

        for repo_path in repo_files:
            if repo_prefix and not repo_path.startswith(repo_prefix):
                continue
            local_path = merged_root / repo_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=repo_path,
                revision=branch,
                token=resolved_token,
            )
            shutil.copy2(downloaded_path, local_path)

        current_submission_targets: list[tuple[Path, Path]] = []
        for submission_dir in normalized_submission_dirs:
            current_submission_target = merged_source_dir / submission_dir.name
            shutil.copytree(submission_dir, current_submission_target, dirs_exist_ok=True)
            current_submission_targets.append((submission_dir, current_submission_target))

        aggregate_rc = aggregate_to_website(
            layout=layout,
            source_dir=merged_source_dir,
            output_dir=aggregate_output_dir,
            execute=True,
        )
        if aggregate_rc != 0:
            return aggregate_rc

        aggregate_files = [
            "leaderboard_single.json",
            "leaderboard_multi.json",
            "leaderboard_compare.json",
            "last_updated.json",
        ]
        operations: list[CommitOperationAdd] = []
        planned_paths: list[str] = []

        for file_name in aggregate_files:
            local_file = aggregate_output_dir / file_name
            if not local_file.is_file():
                print(f"missing aggregated output: {local_file}", file=sys.stderr)
                return 2
            operations.append(
                CommitOperationAdd(path_in_repo=file_name, path_or_fileobj=local_file)
            )
            planned_paths.append(file_name)

        for submission_dir, current_submission_target in current_submission_targets:
            for local_file in sorted(current_submission_target.rglob("*")):
                if not local_file.is_file():
                    continue
                relative_path = local_file.relative_to(current_submission_target).as_posix()
                repo_path = "/".join(
                    part
                    for part in [normalized_prefix, submission_dir.name, relative_path]
                    if part
                )
                operations.append(
                    CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_file)
                )
                planned_paths.append(repo_path)

        if dry_run:
            print(f"[dry-run] Would upload {len(planned_paths)} file(s) to {repo_id}@{branch}:")
            for repo_path in planned_paths:
                print(f"  {repo_path}")
            return 0

        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except Exception:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

        _create_commit_on_branch(
            api,
            repo_id=repo_id,
            repo_type="dataset",
            branch=branch,
            operations=operations,
            commit_message=commit_message,
        )
        return 0
