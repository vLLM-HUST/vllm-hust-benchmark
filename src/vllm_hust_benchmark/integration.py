from __future__ import annotations

import json
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
from vllm_hust_benchmark.registry import get_scenario
from vllm_hust_benchmark.submission_artifacts import iter_submission_artifact_paths
from vllm_hust_benchmark.submission_artifacts import normalize_submission_artifacts_in_tree

FLAG_PATTERN = re.compile(r"^\s+--([a-z0-9][a-z0-9-_]*)\b", re.MULTILINE)
DEFAULT_RUNTIME_ENGINE = "vllm-hust"
DEFAULT_LOCAL_SERVER_HOST = "127.0.0.1"
DEFAULT_LOCAL_SERVER_PORT = 8000
DEFAULT_READY_TIMEOUT_SECONDS = 180.0
BASELINE_STATUS_OFFICIAL_COVERED = "official-covered"
BASELINE_STATUS_PENDING = "pending-baseline"
BASELINE_STATUS_NONE = "no-baseline-declared"
VALID_BASELINE_STATUSES = frozenset(
    {
        BASELINE_STATUS_OFFICIAL_COVERED,
        BASELINE_STATUS_PENDING,
        BASELINE_STATUS_NONE,
    }
)
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
        os.environ.get("VLLM_HUST_BENCHMARK_REPO")
        or workspace_root / "vllm-hust-benchmark"
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
    tests_dir = (
        layout.vllm_hust_repo / ".buildkite" / "performance-benchmarks" / "tests"
    )
    if not tests_dir.is_dir():
        raise ValueError(
            f"vllm-hust performance benchmark tests not found: {tests_dir}"
        )
    suite_script = (
        layout.vllm_hust_repo
        / ".buildkite"
        / "performance-benchmarks"
        / "scripts"
        / "run-performance-benchmarks.sh"
    )
    if not suite_script.is_file():
        raise ValueError(
            f"vllm-hust performance benchmark suite not found: {suite_script}"
        )


def _is_usable_executable(executable_path: str) -> bool:
    path = Path(executable_path)
    if not path.is_file() or not os.access(path, os.X_OK):
        return False

    try:
        first_line = path.read_bytes().splitlines()[0] if path.stat().st_size else b""
    except OSError:
        return False

    if not first_line.startswith(b"#!"):
        return True

    shebang = first_line[2:].decode("utf-8", errors="ignore").strip()
    if not shebang:
        return True

    parts = shlex.split(shebang)
    if not parts:
        return True

    interpreter = parts[0]
    if Path(interpreter).name == "env":
        if len(parts) < 2:
            return False
        return shutil.which(parts[1]) is not None
    return Path(interpreter).is_file() and os.access(interpreter, os.X_OK)


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
        if executable_path and _is_usable_executable(executable_path):
            return [executable_path, *command_args]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main", *command_args]


def _resolve_flag_discovery_cwd(
    layout: RepoLayout,
    *,
    runtime_engine: str = DEFAULT_RUNTIME_ENGINE,
    runtime_repo: str | None = None,
) -> Path:
    candidate = (
        Path(runtime_repo).resolve()
        if runtime_repo
        else resolve_runtime_repo(layout, runtime_engine)
    )
    if candidate.is_dir():
        return candidate
    if layout.benchmark_repo.is_dir():
        return layout.benchmark_repo
    return Path.cwd().resolve()


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
    cwd = _resolve_flag_discovery_cwd(
        layout,
        runtime_engine=runtime_engine,
        runtime_repo=runtime_repo,
    )
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
        match.group(1).replace("-", "_") for match in FLAG_PATTERN.finditer(help_text)
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
    runtime_repo = validate_runtime_repo(
        layout, runtime_engine, require_benchmarks=True
    )
    script_path = runtime_repo / "benchmarks" / script_name
    if not script_path.is_file():
        raise ValueError(f"benchmark script not found: {script_path}")

    if script_path.suffix == ".py":
        return [sys.executable, str(script_path), *script_args]
    if script_path.suffix == ".sh":
        return ["bash", str(script_path), *script_args]
    return [str(script_path), *script_args]


def build_performance_suite_command(layout: RepoLayout) -> list[str]:
    suite_script = (
        layout.vllm_hust_repo
        / ".buildkite"
        / "performance-benchmarks"
        / "scripts"
        / "run-performance-benchmarks.sh"
    )
    return ["bash", str(suite_script)]


def _resolve_stable_repo(worktree_path: Path) -> Path:
    """Resolve the stable (non-worktree) vllm-hust repo from a worktree path.

    A git worktree's .git is a file (not a directory) pointing to the actual
    .git directory: 'gitdir: /path/to/main/.git/worktrees/<name>'.
    We read this to find the main repo's .git/worktrees/ and then navigate
    up to the main repo root.
    """
    git_file = worktree_path / ".git"
    if not git_file.is_file():
        return worktree_path  # Not a worktree, use as-is
    content = git_file.read_text().strip()
    if not content.startswith("gitdir: "):
        return worktree_path  # Not a worktree .git file
    # e.g. "gitdir: /path/to/main/.git/worktrees/vllm-hust--abc123"
    gitdir = Path(content.split(": ", 1)[1].strip())
    # The gitdir is .git/worktrees/<name>, go up two levels to main repo root
    main_git_dir = gitdir.parent.parent  # -> /path/to/main/.git
    main_repo = main_git_dir.parent  # -> /path/to/main
    if main_repo.is_dir():
        return main_repo
    return worktree_path  # Fallback


def build_ascend_benchmark_ci_command(layout: RepoLayout) -> list[str]:
    # CI script is an infrastructure file that only exists on the stable branch,
    # not in arbitrary commit worktrees. Detect if we are inside a worktree and
    # resolve to the stable main repo to avoid:
    #   "vllm-hust Ascend benchmark CI script not found: .../worktrees/vllm-hust--<sha>/..."
    repo = _resolve_stable_repo(layout.vllm_hust_repo)
    ci_script = (
        repo
        / ".github"
        / "workflows"
        / "scripts"
        / "run_ascend_benchmark_ci.sh"
    )
    if not ci_script.is_file():
        raise ValueError(f"vllm-hust Ascend benchmark CI script not found: {ci_script}")
    return ["bash", str(ci_script)]


def _format_env_prefix(env: Mapping[str, object] | None) -> str:
    if not env:
        return ""
    return (
        " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items()) + " "
    )


def _build_effective_env(
    cwd: Path,
    env: Mapping[str, object] | None,
) -> dict[str, str]:
    effective_env = {key: str(value) for key, value in os.environ.items()}
    if env:
        effective_env.update({key: str(value) for key, value in env.items()})

    cwd_str = str(cwd)
    existing_pythonpath = effective_env.get("PYTHONPATH", "")
    pythonpath_entries = [
        entry for entry in existing_pythonpath.split(os.pathsep) if entry
    ]
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


def _load_snapshot_json(path: Path) -> Any:
    if not path.is_file():
        raise ValueError(f"missing aggregated output: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in aggregated output {path}: {exc}") from exc


def _normalize_engine(entry: Mapping[str, Any]) -> str:
    return (
        str(entry.get("engine") or entry.get("metadata", {}).get("engine") or "unknown")
        .strip()
        .lower()
    )


def _extract_workload_name(entry: Mapping[str, Any]) -> str:
    workload = entry.get("workload")
    if isinstance(workload, Mapping) and workload.get("name"):
        return str(workload["name"])
    return str(
        entry.get("workload_name")
        or entry.get("metadata", {}).get("workload")
        or "Other"
    )


def _normalize_baseline_engine(value: Any) -> str:
    return str(value or "").strip().lower()


def _build_entry_baseline_coverage_key(
    entry: Mapping[str, Any], *, engine: str
) -> str:
    model = entry.get("model") if isinstance(entry.get("model"), Mapping) else {}
    hardware = (
        entry.get("hardware") if isinstance(entry.get("hardware"), Mapping) else {}
    )
    return _build_baseline_coverage_key(
        engine=engine,
        model=str(model.get("name") or "unknown-model"),
        hardware=str(hardware.get("chip_model") or "unknown-hardware"),
        workload=_extract_workload_name(entry),
        config_type=str(entry.get("config_type") or "unknown-config"),
    )


def _derive_accountable_scope_baseline_metadata(
    entry: Mapping[str, Any],
    *,
    official_coverage_keys: set[str] | None = None,
) -> dict[str, str]:
    constraints = entry.get("constraints")
    accountable = (
        constraints.get("accountable_scope") if isinstance(constraints, Mapping) else {}
    )
    accountable = accountable if isinstance(accountable, Mapping) else {}

    official_engine = _normalize_baseline_engine(accountable.get("baseline_engine"))
    declared_engine = _normalize_baseline_engine(
        accountable.get("declared_baseline_engine") or official_engine
    )
    baseline_status = str(accountable.get("baseline_status") or "").strip()

    if official_coverage_keys is not None:
        if declared_engine:
            baseline_key = _build_entry_baseline_coverage_key(
                entry, engine=declared_engine
            )
            if baseline_key in official_coverage_keys:
                baseline_status = BASELINE_STATUS_OFFICIAL_COVERED
                official_engine = declared_engine
            else:
                baseline_status = BASELINE_STATUS_PENDING
                official_engine = ""
        else:
            baseline_status = BASELINE_STATUS_NONE
            official_engine = ""
    else:
        if baseline_status not in VALID_BASELINE_STATUSES:
            if official_engine:
                baseline_status = BASELINE_STATUS_OFFICIAL_COVERED
            elif declared_engine:
                baseline_status = BASELINE_STATUS_PENDING
            else:
                baseline_status = BASELINE_STATUS_NONE

        if baseline_status == BASELINE_STATUS_OFFICIAL_COVERED:
            official_engine = official_engine or declared_engine
        else:
            official_engine = ""

    return {
        "baseline_engine": official_engine,
        "declared_baseline_engine": declared_engine,
        "baseline_status": baseline_status,
        "scope_baseline_engine": declared_engine or official_engine or "unknown-baseline",
    }


def _normalize_accountable_scope_baseline_metadata(
    entry: dict[str, Any],
    *,
    official_coverage_keys: set[str] | None = None,
) -> dict[str, Any] | None:
    constraints = entry.get("constraints")
    constraints = constraints if isinstance(constraints, dict) else None
    if constraints is None:
        return None

    accountable_scope = constraints.get("accountable_scope")
    accountable_scope = accountable_scope if isinstance(accountable_scope, dict) else None
    if accountable_scope is None:
        return None

    baseline_metadata = _derive_accountable_scope_baseline_metadata(
        entry, official_coverage_keys=official_coverage_keys
    )
    accountable_scope["baseline_engine"] = baseline_metadata["baseline_engine"]
    accountable_scope["declared_baseline_engine"] = baseline_metadata[
        "declared_baseline_engine"
    ]
    accountable_scope["baseline_status"] = baseline_metadata["baseline_status"]
    return accountable_scope


def _build_hard_constraint_scope_key(entry: Mapping[str, Any]) -> str:
    baseline_metadata = _derive_accountable_scope_baseline_metadata(entry)
    constraints = entry.get("constraints")
    accountable = (
        constraints.get("accountable_scope") if isinstance(constraints, Mapping) else {}
    )
    accountable = accountable if isinstance(accountable, Mapping) else {}
    model = entry.get("model") if isinstance(entry.get("model"), Mapping) else {}
    hardware = (
        entry.get("hardware") if isinstance(entry.get("hardware"), Mapping) else {}
    )
    return "|".join(
        [
            _normalize_engine(entry),
            str(model.get("name") or "unknown-model"),
            str(hardware.get("chip_model") or "unknown-hardware"),
            _extract_workload_name(entry),
            str(entry.get("config_type") or "unknown-config"),
            str(
                accountable.get("representative_business_scenario")
                or "unknown-business-scenario"
            ),
            baseline_metadata["scope_baseline_engine"],
        ]
    )


def _build_baseline_coverage_key(
    *,
    engine: str,
    model: str,
    hardware: str,
    workload: str,
    config_type: str,
) -> str:
    return "|".join([engine, model, hardware, workload, config_type])


def _classify_config_type(*, chip_count: int, node_count: int) -> str:
    if node_count > 1:
        return "multi_node"
    if chip_count > 1:
        return "multi_gpu"
    return "single_gpu"


def _load_official_baseline_coverage_keys(layout: RepoLayout) -> set[str]:
    official_specs_dir = layout.benchmark_repo / "docs" / "official-baselines"
    if not official_specs_dir.is_dir():
        return set()

    declared_coverage_keys: set[str] = set()
    for spec_path in sorted(official_specs_dir.glob("*.json")):
        if spec_path.name.endswith("constraints.stub.json"):
            continue

        try:
            payload = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, Mapping):
            continue

        export_payload = payload.get("export")
        export_payload = export_payload if isinstance(export_payload, Mapping) else {}
        baseline_engine = str(export_payload.get("baseline_engine") or "").strip().lower()
        if not baseline_engine:
            continue

        scenario_name = str(payload.get("scenario") or "").strip()
        if not scenario_name:
            continue
        try:
            scenario = get_scenario(scenario_name)
        except KeyError:
            continue

        workload = scenario.leaderboard.get("workload_name") or scenario_name
        chip_count = int(payload.get("chip_count") or 1)
        node_count = int(payload.get("node_count") or 1)
        declared_coverage_keys.add(
            _build_baseline_coverage_key(
                engine=baseline_engine,
                model=str(payload.get("model") or "unknown-model"),
                hardware=str(payload.get("hardware_chip_model") or "unknown-hardware"),
                workload=str(workload),
                config_type=_classify_config_type(
                    chip_count=chip_count,
                    node_count=node_count,
                ),
            )
        )

    if not declared_coverage_keys:
        return set()

    submissions_root = layout.benchmark_repo / "submissions"
    if not submissions_root.is_dir():
        return set()

    published_coverage_keys: set[str] = set()
    for artifact_path in sorted(submissions_root.rglob("run_leaderboard.json")):
        if not artifact_path.parent.name.startswith("official-ascend-"):
            continue

        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, Mapping):
            continue

        coverage_key = _build_entry_baseline_coverage_key(
            payload,
            engine=_normalize_engine(payload),
        )
        if coverage_key in declared_coverage_keys:
            published_coverage_keys.add(coverage_key)

    return published_coverage_keys


def _normalize_submission_baseline_metadata(
    source_dir: Path,
    *,
    official_coverage_keys: set[str],
) -> None:
    for artifact_path in iter_submission_artifact_paths(source_dir):
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(payload, dict):
            continue

        accountable_scope = _normalize_accountable_scope_baseline_metadata(
            payload, official_coverage_keys=official_coverage_keys
        )
        if accountable_scope is None:
            continue

        artifact_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )


def _get_same_spec_payload(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = entry.get("same_spec")
    return payload if isinstance(payload, Mapping) else {}


def _get_same_spec_id(entry: Mapping[str, Any]) -> str | None:
    spec_id = str(_get_same_spec_payload(entry).get("spec_id") or "").strip()
    return spec_id or None


def _get_same_spec_hash(entry: Mapping[str, Any]) -> str | None:
    spec_hash = str(
        _get_same_spec_payload(entry).get("resolved_spec_hash") or ""
    ).strip()
    return spec_hash or None


def _build_compare_setting_signature(entry: Mapping[str, Any]) -> str:
    same_spec_hash = _get_same_spec_hash(entry)
    if same_spec_hash:
        return same_spec_hash

    workload = entry.get("workload") if isinstance(entry.get("workload"), Mapping) else {}
    same_spec = _get_same_spec_payload(entry)
    server = same_spec.get("resolved_server_parameters")
    server = server if isinstance(server, Mapping) else {}
    client = same_spec.get("resolved_client_parameters")
    client = client if isinstance(client, Mapping) else {}

    input_length = workload.get("input_length")
    output_length = workload.get("output_length")
    tensor_parallel = server.get("tensor_parallel_size")
    pipeline_parallel = server.get("pipeline_parallel_size")
    dtype = server.get("dtype")
    request_rate = client.get("request_rate")
    return "|".join(
        [
            str(input_length if input_length is not None else "unknown-input"),
            str(output_length if output_length is not None else "unknown-output"),
            str(tensor_parallel if tensor_parallel is not None else "unknown-tp"),
            str(pipeline_parallel if pipeline_parallel is not None else "unknown-pp"),
            str(dtype or "unknown-dtype"),
            str(request_rate if request_rate is not None else "unknown-rps"),
        ]
    )


def _build_compare_scope_key_debug(entry: Mapping[str, Any]) -> str:
    model = str((entry.get("model") or {}).get("name") or "unknown-model")
    hardware = str(
        (entry.get("hardware") or {}).get("chip_model") or "unknown-hardware"
    )
    precision = str((entry.get("model") or {}).get("precision") or "unknown-precision")
    workload = _extract_workload_name(entry)
    config_type = str(entry.get("config_type") or "unknown-config")
    chip_count = int((entry.get("hardware") or {}).get("chip_count") or 0)
    node_count = int((entry.get("cluster") or {}).get("node_count") or 1)
    setting_signature = _build_compare_setting_signature(entry)
    return "|".join(
        [
            model,
            hardware,
            precision,
            workload,
            config_type,
            str(chip_count),
            str(node_count),
            setting_signature,
        ]
    )


def _normalize_goal_model_name(model_name: Any) -> str:
    raw_name = str(model_name or "unknown-model").strip()
    if not raw_name:
        return "unknown-model"
    if "/" not in raw_name:
        return raw_name
    return raw_name.rsplit("/", maxsplit=1)[-1] or raw_name


def _build_goal_scope_key_debug(entry: Mapping[str, Any]) -> str:
    model = _normalize_goal_model_name((entry.get("model") or {}).get("name"))
    hardware = str(
        (entry.get("hardware") or {}).get("chip_model") or "unknown-hardware"
    )
    precision = str((entry.get("model") or {}).get("precision") or "unknown-precision")
    workload = _extract_workload_name(entry)
    config_type = str(entry.get("config_type") or "unknown-config")
    chip_count = int((entry.get("hardware") or {}).get("chip_count") or 0)
    node_count = int((entry.get("cluster") or {}).get("node_count") or 1)
    setting_signature = _build_compare_setting_signature(entry)
    return "|".join(
        [
            model,
            hardware,
            precision,
            workload,
            config_type,
            str(chip_count),
            str(node_count),
            setting_signature,
        ]
    )


def _is_goal_baseline_entry_debug(entry: Mapping[str, Any]) -> bool:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
    engine = str(entry.get("engine") or metadata.get("engine") or "unknown").strip().lower()
    engine_version = str(
        entry.get("engine_version") or metadata.get("engine_version") or ""
    ).strip()
    github_repository = str(metadata.get("github_repository") or "").strip().lower()
    return (
        engine == "vllm"
        and engine_version.startswith("0.11.0")
        and github_repository == "vllm-project/vllm-ascend"
    )


def _print_aggregated_compare_diagnostics(data_dir: Path) -> None:
    try:
        single = _load_snapshot_json(data_dir / "leaderboard_single.json")
        multi = _load_snapshot_json(data_dir / "leaderboard_multi.json")
        compare = _load_snapshot_json(data_dir / "leaderboard_compare.json")
    except ValueError as exc:
        print(f"compare debug unavailable: {exc}", file=sys.stderr)
        return

    entries = [
        *(single if isinstance(single, list) else []),
        *(multi if isinstance(multi, list) else []),
    ]
    print("aggregated compare debug: entries", file=sys.stderr)
    for entry in sorted(
        entries,
        key=lambda item: (
            _normalize_engine(item),
            str((item.get("model") or {}).get("name") or ""),
            _extract_workload_name(item),
            str((item.get("metadata") or {}).get("submitted_at") or ""),
        ),
    ):
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), Mapping) else {}
        accountable = {}
        constraints = entry.get("constraints")
        if isinstance(constraints, Mapping) and isinstance(
            constraints.get("accountable_scope"), Mapping
        ):
            accountable = constraints.get("accountable_scope")
        print(
            "  "
            + " | ".join(
                [
                    f"engine={_normalize_engine(entry)}",
                    f"engine_version={str(entry.get('engine_version') or metadata.get('engine_version') or '')}",
                    f"model={str((entry.get('model') or {}).get('name') or '')}",
                    f"workload={_extract_workload_name(entry)}",
                    f"config_type={str(entry.get('config_type') or '')}",
                    f"chip_count={int((entry.get('hardware') or {}).get('chip_count') or 0)}",
                    f"node_count={int((entry.get('cluster') or {}).get('node_count') or 1)}",
                    f"baseline_engine={str(accountable.get('baseline_engine') or '')}",
                    f"spec_id={_get_same_spec_id(entry) or ''}",
                    f"spec_hash={_get_same_spec_hash(entry) or ''}",
                    f"github_repository={str(metadata.get('github_repository') or '')}",
                ]
            ),
            file=sys.stderr,
        )

    compare_grouped: dict[str, list[Mapping[str, Any]]] = {}
    goal_grouped: dict[str, list[Mapping[str, Any]]] = {}
    for entry in entries:
        compare_grouped.setdefault(_build_compare_scope_key_debug(entry), []).append(entry)
        goal_grouped.setdefault(_build_goal_scope_key_debug(entry), []).append(entry)

    print("aggregated compare debug: compare-scope candidates", file=sys.stderr)
    for scope_key, scope_entries in sorted(compare_grouped.items()):
        engines = sorted({_normalize_engine(entry) for entry in scope_entries})
        if len(engines) < 2:
            continue
        print(f"  scope={scope_key}", file=sys.stderr)
        for entry in scope_entries:
            print(
                "    "
                + " | ".join(
                    [
                        f"engine={_normalize_engine(entry)}",
                        f"model={str((entry.get('model') or {}).get('name') or '')}",
                        f"spec_hash={_get_same_spec_hash(entry) or ''}",
                        f"submitted_at={str((entry.get('metadata') or {}).get('submitted_at') or '')}",
                    ]
                ),
                file=sys.stderr,
            )

    print("aggregated compare debug: goal-scope candidates", file=sys.stderr)
    for scope_key, scope_entries in sorted(goal_grouped.items()):
        current_entries = [entry for entry in scope_entries if _normalize_engine(entry) == "vllm-hust"]
        baseline_entries = [entry for entry in scope_entries if _is_goal_baseline_entry_debug(entry)]
        if not current_entries and not baseline_entries:
            continue
        print(
            f"  scope={scope_key} | current={len(current_entries)} | baseline={len(baseline_entries)}",
            file=sys.stderr,
        )
        for entry in current_entries + baseline_entries:
            print(
                "    "
                + " | ".join(
                    [
                        f"engine={_normalize_engine(entry)}",
                        f"model={str((entry.get('model') or {}).get('name') or '')}",
                        f"spec_hash={_get_same_spec_hash(entry) or ''}",
                        f"github_repository={str((entry.get('metadata') or {}).get('github_repository') or '')}",
                    ]
                ),
                file=sys.stderr,
            )

    hard_constraints = compare.get("hard_constraints") if isinstance(compare, Mapping) else {}
    hard_constraints = hard_constraints if isinstance(hard_constraints, Mapping) else {}
    scopes = hard_constraints.get("scopes") if isinstance(hard_constraints.get("scopes"), list) else []
    goal_progress = compare.get("goal_progress") if isinstance(compare, Mapping) else {}
    goal_progress = goal_progress if isinstance(goal_progress, Mapping) else {}
    print(
        "aggregated compare debug: summary"
        f" | compare_group_count={int(compare.get('group_count') or 0) if isinstance(compare, Mapping) else 0}"
        f" | goal_pair_count={len(goal_progress.get('pairs') or []) if isinstance(goal_progress.get('pairs'), list) else 0}"
        f" | hard_constraint_scope_count={len(scopes)}",
        file=sys.stderr,
    )


def validate_aggregated_leaderboard_outputs(
    data_dir: Path,
    *,
    official_coverage_keys: set[str] | None = None,
) -> None:
    single = _load_snapshot_json(data_dir / "leaderboard_single.json")
    multi = _load_snapshot_json(data_dir / "leaderboard_multi.json")
    compare = _load_snapshot_json(data_dir / "leaderboard_compare.json")

    tab_engines: dict[str, set[str]] = {
        "single-chip": set(),
        "multi-chip": set(),
        "multi-node": set(),
    }
    for entry in single if isinstance(single, list) else []:
        tab_engines["single-chip"].add(_normalize_engine(entry))
    for entry in multi if isinstance(multi, list) else []:
        cluster = entry.get("cluster") if isinstance(entry, Mapping) else {}
        cluster = cluster if isinstance(cluster, Mapping) else {}
        node_count = cluster.get("node_count") or 1
        tab = "multi-node" if node_count > 1 else "multi-chip"
        tab_engines[tab].add(_normalize_engine(entry))

    populated_tabs = {
        tab: sorted(engines) for tab, engines in tab_engines.items() if engines
    }
    if populated_tabs and all(
        engines == ["vllm-hust"] for engines in populated_tabs.values()
    ):
        raise ValueError(
            "invalid aggregated leaderboard outputs: all populated tabs contain only vllm-hust entries "
            f"({populated_tabs})"
        )

    hard_constraints = (
        compare.get("hard_constraints") if isinstance(compare, Mapping) else {}
    )
    hard_constraints = hard_constraints if isinstance(hard_constraints, Mapping) else {}
    hard_constraint_scopes = (
        hard_constraints.get("scopes")
        if isinstance(hard_constraints.get("scopes"), list)
        else []
    )
    groups = (
        compare.get("groups")
        if isinstance(compare, Mapping) and isinstance(compare.get("groups"), list)
        else []
    )
    goal_progress = compare.get("goal_progress") if isinstance(compare, Mapping) else {}
    goal_progress = goal_progress if isinstance(goal_progress, Mapping) else {}
    goal_pairs = (
        goal_progress.get("pairs")
        if isinstance(goal_progress.get("pairs"), list)
        else []
    )

    scope_keys = {
        _build_hard_constraint_scope_key(entry)
        for entry in [
            *(single if isinstance(single, list) else []),
            *(multi if isinstance(multi, list) else []),
        ]
        if _normalize_engine(entry) == "vllm-hust"
    }
    missing_scope_keys = [
        scope.get("scope_key")
        for scope in hard_constraint_scopes
        if str(scope.get("scope_key") or "") not in scope_keys
    ]
    if missing_scope_keys:
        raise ValueError(
            "invalid aggregated leaderboard outputs: hard-constraint scope keys are missing from main snapshots "
            f"({missing_scope_keys})"
        )


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
        validate_aggregated_leaderboard_outputs(data_dir)
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
    return (
        token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def _upload_existing_snapshots(
    *,
    api,
    repo_id: str,
    branch: str,
    aggregate_output_dir: Path,
    commit_message: str,
    dry_run: bool,
) -> int:
    """Directly upload existing leaderboard snapshots without re-aggregation.

    Used by --skip-aggregation to sync HF write side to match a known-good
    snapshot set (e.g. the read-side data) without re-running aggregation
    from submissions.
    """
    try:
        from huggingface_hub import CommitOperationAdd
        from vllm_hust_benchmark.hf_publisher import _create_commit_on_branch
    except ImportError:
        print(
            "huggingface_hub is required for HF submission sync. Install with: "
            "pip install 'vllm-hust-benchmark[publish]'",
            file=sys.stderr,
        )
        return 2

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

    if dry_run:
        print(
            f"[dry-run] Would upload {len(planned_paths)} snapshot file(s) to {repo_id}@{branch}:"
        )
        for repo_path in planned_paths:
            print(f"  {repo_path}")
        return 0

    print(f"Uploading {len(planned_paths)} snapshot file(s) to {repo_id}@{branch}...")

    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception:
        api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
        )

    _create_commit_on_branch(
        api,
        repo_id=repo_id,
        repo_type="dataset",
        branch=branch,
        operations=operations,
        commit_message=commit_message,
    )
    print("Upload complete.")
    return 0


def sync_submission_to_huggingface(
    *,
    layout: RepoLayout,
    submission_dirs: Path | Sequence[Path] | None,
    aggregate_output_dir: Path,
    repo_id: str,
    token: str | None = None,
    branch: str = "main",
    submissions_prefix: str = "submissions-auto",
    commit_message: str = "chore: sync benchmark submission and leaderboard data",
    dry_run: bool = False,
    allow_existing_only: bool = False,
    skip_aggregation: bool = False,
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

    if submission_dirs is None:
        normalized_submission_dirs: list[Path] = []
    elif isinstance(submission_dirs, Path):
        normalized_submission_dirs = [submission_dirs]
    else:
        normalized_submission_dirs = list(submission_dirs)

    if not normalized_submission_dirs and not allow_existing_only and not skip_aggregation:
        print(
            "at least one submission directory is required unless --existing-only or --skip-aggregation is set",
            file=sys.stderr,
        )
        return 2

    for submission_dir in normalized_submission_dirs:
        if not submission_dir.is_dir():
            print(f"submission directory not found: {submission_dir}", file=sys.stderr)
            return 2

    resolved_token = _resolve_hf_token(token)
    api = HfApi(token=resolved_token)
    normalized_prefix = submissions_prefix.strip("/")
    repo_prefix = f"{normalized_prefix}/" if normalized_prefix else ""

    # When --skip-aggregation is set, directly upload existing snapshots
    # without re-aggregating from submissions.
    if skip_aggregation:
        return _upload_existing_snapshots(
            api=api,
            repo_id=repo_id,
            branch=branch,
            aggregate_output_dir=aggregate_output_dir,
            commit_message=commit_message,
            dry_run=dry_run,
        )

    with tempfile.TemporaryDirectory(prefix="vllm-hust-hf-sync-") as temp_dir:
        temp_root = Path(temp_dir)
        merged_root = temp_root / "merged_submissions"
        merged_source_dir = (
            merged_root / normalized_prefix if normalized_prefix else merged_root
        )
        merged_source_dir.mkdir(parents=True, exist_ok=True)

        try:
            repo_files = api.list_repo_files(
                repo_id=repo_id,
                repo_type="dataset",
                revision=branch,
            )
        except Exception as exc:
            print(
                f"failed to list dataset files from {repo_id}@{branch}: {exc}",
                file=sys.stderr,
            )
            return 2

        prefixed_repo_files = [
            repo_path
            for repo_path in repo_files
            if not repo_prefix or repo_path.startswith(repo_prefix)
        ]
        if allow_existing_only and not prefixed_repo_files:
            print(
                f"no historical submissions found under prefix {normalized_prefix!r} "
                f"in {repo_id}@{branch}",
                file=sys.stderr,
            )
            return 2

        for repo_path in prefixed_repo_files:
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

        for submission_dir in normalized_submission_dirs:
            current_submission_target = merged_source_dir / submission_dir.name
            shutil.copytree(
                submission_dir, current_submission_target, dirs_exist_ok=True
            )

        official_coverage_keys = _load_official_baseline_coverage_keys(layout)

        normalize_submission_artifacts_in_tree(merged_source_dir)
        _normalize_submission_baseline_metadata(
            merged_source_dir,
            official_coverage_keys=official_coverage_keys,
        )

        aggregate_rc = aggregate_to_website(
            layout=layout,
            source_dir=merged_source_dir,
            output_dir=aggregate_output_dir,
            execute=True,
        )
        if aggregate_rc != 0:
            return aggregate_rc

        try:
            validate_aggregated_leaderboard_outputs(
                aggregate_output_dir,
                official_coverage_keys=official_coverage_keys,
            )
        except ValueError as exc:
            _print_aggregated_compare_diagnostics(aggregate_output_dir)
            print(str(exc), file=sys.stderr)
            return 2

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

        for local_file in sorted(merged_source_dir.rglob("*")):
            if not local_file.is_file():
                continue
            relative_path = local_file.relative_to(merged_source_dir).as_posix()
            repo_path = "/".join(
                part for part in [normalized_prefix, relative_path] if part
            )
            operations.append(
                CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_file)
            )
            planned_paths.append(repo_path)

        if dry_run:
            print(
                f"[dry-run] Would upload {len(planned_paths)} file(s) to {repo_id}@{branch}:"
            )
            for repo_path in planned_paths:
                print(f"  {repo_path}")
            return 0

        try:
            api.repo_info(repo_id=repo_id, repo_type="dataset")
        except Exception:
            api.create_repo(
                repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
            )

        _create_commit_on_branch(
            api,
            repo_id=repo_id,
            repo_type="dataset",
            branch=branch,
            operations=operations,
            commit_message=commit_message,
        )
        return 0
