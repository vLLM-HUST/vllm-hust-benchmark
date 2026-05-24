from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from vllm_hust_benchmark.combo_manifest import (
    fingerprint_payload,
    get_runtime_engine,
    load_combo_manifest,
)
from vllm_hust_benchmark.integration import RepoLayout, resolve_repo_layout


def _git_stdout(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed for {repo}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _ensure_git_repo(path: Path, label: str) -> None:
    if not path.joinpath(".git").exists():
        raise ValueError(f"{label} is not a git checkout: {path}")


def _resolve_source_repo_paths(layout: RepoLayout) -> dict[str, Path]:
    vllm_ascend_hust_repo = layout.vllm_ascend_hust_repo or (
        layout.workspace_root / "vllm-ascend-hust"
    ).resolve()
    return {
        "vllm_hust": layout.vllm_hust_repo,
        "vllm_ascend_hust": vllm_ascend_hust_repo,
    }


def _materialize_git_ref(source_repo: Path, destination: Path, ref: str) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(source_repo),
            "worktree",
            "add",
            "--force",
            "--detach",
            str(destination),
            ref,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"Unable to materialize {source_repo}@{ref} at {destination}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )


def _remove_worktree(source_repo: Path, worktree_path: Path) -> None:
    subprocess.run(
        ["git", "-C", str(source_repo), "worktree", "remove", "--force", str(worktree_path)],
        check=False,
        capture_output=True,
        text=True,
    )


def materialize_combo_manifest(
    *,
    layout: RepoLayout,
    manifest: Mapping[str, Any],
    runtime_root: Path,
) -> dict[str, dict[str, str]]:
    source_paths = _resolve_source_repo_paths(layout)
    for repo_key, source_repo in source_paths.items():
        _ensure_git_repo(source_repo, repo_key)

    source_combo = manifest["source_combo"]
    materialized: dict[str, dict[str, str]] = {}
    for repo_key in ("vllm_hust", "vllm_ascend_hust"):
        repo_payload = source_combo[repo_key]
        source_repo = source_paths[repo_key]
        destination = runtime_root / repo_key.replace("_", "-")
        _materialize_git_ref(source_repo, destination, str(repo_payload["ref"]))
        materialized[repo_key] = {
            "source_repo": str(source_repo),
            "path": str(destination),
            "ref": str(repo_payload["ref"]),
            "repo": str(repo_payload["repo"]),
            "commit": _git_stdout(destination, "rev-parse", "HEAD"),
        }
    return materialized


def _build_script_env(
    *,
    layout: RepoLayout,
    manifest: Mapping[str, Any],
    materialized: Mapping[str, Mapping[str, str]],
    result_root: Path,
) -> dict[str, str]:
    benchmark_config = manifest["benchmark_config"]
    server_config = benchmark_config["server"]
    latency_config = benchmark_config["latency"]
    throughput_config = benchmark_config["throughput"]
    serve_config = benchmark_config["serve"]
    source_combo = manifest["source_combo"]

    raw_result_file = result_root / "raw_benchmark.json"
    server_log = result_root / "server.log"
    env = {
        "WORKSPACE_ROOT": str(result_root.parent),
        "VLLM_HUST_REPO": materialized["vllm_hust"]["path"],
        "VLLM_ASCEND_HUST_REPO": materialized["vllm_ascend_hust"]["path"],
        "VLLM_HUST_BENCHMARK_REPO": str(layout.benchmark_repo),
        "VLLM_HUST_WEBSITE_REPO": str(layout.website_repo),
        "RUN_ID": result_root.name,
        "RESULT_ROOT": str(result_root),
        "RAW_RESULT_FILE": str(raw_result_file),
        "SERVER_LOG": str(server_log),
        "MODEL_NAME": str(benchmark_config["model"]),
        "LOAD_FORMAT": str(benchmark_config.get("load_format") or ""),
        "DTYPE": str(benchmark_config["dtype"]),
        "MAX_MODEL_LEN": str(server_config["max_model_len"]),
        "MAX_NUM_SEQS": str(server_config["max_num_seqs"]),
        "BENCH_SCENARIO": str(serve_config["scenario"]),
        "BENCH_NUM_PROMPTS": str(serve_config["num_prompts"]),
        "BENCH_REQUEST_RATE": str(serve_config["request_rate"]),
        "BENCH_MAX_CONCURRENCY": str(serve_config["max_concurrency"]),
        "BENCH_RANDOM_INPUT_LEN": str(throughput_config["input_len"]),
        "BENCH_RANDOM_OUTPUT_LEN": str(throughput_config["output_len"]),
        "BENCH_RANDOM_BATCH_SIZE": str(throughput_config["random_batch_size"]),
        "BENCH_INPUT_LEN": str(latency_config["input_len"]),
        "BENCH_OUTPUT_LEN": str(latency_config["output_len"]),
    }
    if os.environ.get("RUNNER_CLASS"):
        env["RUNNER_CLASS"] = os.environ["RUNNER_CLASS"]
    if os.environ.get("SOC_VERSION"):
        env["SOC_VERSION"] = os.environ["SOC_VERSION"]

    if source_combo["trigger_repo"] == "vllm-ascend-hust":
        env.update(
            {
                "ASCEND_HUST_TARGET_REPOSITORY": str(
                    source_combo["vllm_ascend_hust"]["repo"]
                ),
                "ASCEND_HUST_TARGET_REF": str(source_combo["vllm_ascend_hust"]["ref"]),
                "ASCEND_HUST_TARGET_SHA": str(
                    materialized["vllm_ascend_hust"]["commit"]
                ),
                "VLLM_HUST_REF": str(source_combo["vllm_hust"]["ref"]),
            }
        )
    return env


def _format_env_prefix(env: Mapping[str, str]) -> str:
    if not env:
        return ""
    return " ".join(f"{key}={shlex.quote(str(value))}" for key, value in env.items()) + " "


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _to_float(value: Any, *, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_perf_smoke_result(
    *,
    manifest: Mapping[str, Any],
    materialized: Mapping[str, Mapping[str, str]],
    raw_result_file: Path,
    server_log: Path,
) -> dict[str, Any]:
    raw = _read_json(raw_result_file)
    benchmark_config = manifest["benchmark_config"]
    num_prompts = int(raw.get("num_prompts") or benchmark_config["serve"]["num_prompts"])
    failed_requests = int(raw.get("failed") or 0)
    error_rate = failed_requests / num_prompts if num_prompts > 0 else 0.0
    source_combo = json.loads(json.dumps(manifest["source_combo"]))
    source_combo["vllm_hust"]["commit"] = materialized["vllm_hust"]["commit"]
    source_combo["vllm_ascend_hust"]["commit"] = materialized["vllm_ascend_hust"]["commit"]
    source_combo_fingerprint = fingerprint_payload(source_combo)

    return {
        "schema_version": manifest["schema_version"],
        "benchmark_profile": manifest["benchmark_profile"],
        "benchmark_config_fingerprint": manifest["benchmark_config_fingerprint"],
        "source_combo_fingerprint": source_combo_fingerprint,
        "source_combo": source_combo,
        "benchmark_config": benchmark_config,
        "model": benchmark_config["model"],
        "load_format": benchmark_config.get("load_format"),
        "dtype": benchmark_config["dtype"],
        "tensor_parallel_size": benchmark_config["tensor_parallel_size"],
        "runner_class": os.environ.get("RUNNER_CLASS", "unknown-runner"),
        "soc_version": os.environ.get("SOC_VERSION", "unknown-soc"),
        "serve": {
            "mean_ttft_ms": _to_float(raw.get("mean_ttft_ms")),
            "mean_tpot_ms": _to_float(raw.get("mean_tpot_ms") or raw.get("mean_itl_ms")),
            "request_throughput_rps": _to_float(raw.get("request_throughput")),
            "output_throughput_tps": _to_float(raw.get("output_throughput")),
            "error_rate": error_rate,
            "failed_requests": failed_requests,
        },
        "throughput": {
            "tokens_per_second": _to_float(
                raw.get("total_token_throughput") or raw.get("output_throughput")
            )
        },
        "latency": {
            "mean_ms": _to_float(raw.get("mean_ttft_ms")),
            "p50_ms": _to_float(raw.get("median_ttft_ms")),
        },
        "raw_artifacts": {
            "raw_benchmark_json": str(raw_result_file),
            "server_log": str(server_log),
        },
    }


def run_ascend_perf_smoke(
    *,
    layout: RepoLayout,
    manifest_path: Path,
    output_path: Path,
    result_root: Path,
    execute: bool,
    runtime_root: Path | None = None,
) -> int:
    manifest = load_combo_manifest(manifest_path)
    runtime_engine = get_runtime_engine(manifest)
    runtime_root_parent = result_root.parent
    runtime_root_parent.mkdir(parents=True, exist_ok=True)
    runtime_root_path = runtime_root or Path(
        tempfile.mkdtemp(prefix="combo-perf-smoke-", dir=str(runtime_root_parent))
    )
    script_preview = "run_ascend_benchmark_ci.sh"

    if not execute:
        print(
            f"combo_manifest={manifest_path} runtime_engine={runtime_engine} "
            f"result_root={result_root} output={output_path} script={script_preview}"
        )
        return 0

    result_root.mkdir(parents=True, exist_ok=True)
    materialized: dict[str, dict[str, str]] = {}
    source_paths = _resolve_source_repo_paths(layout)
    try:
        materialized = materialize_combo_manifest(
            layout=layout,
            manifest=manifest,
            runtime_root=runtime_root_path,
        )
        trigger_key = runtime_engine.replace("-", "_")
        trigger_repo_path = Path(materialized[trigger_key]["path"])
        command = [
            "bash",
            str(
                trigger_repo_path
                / ".github"
                / "workflows"
                / "scripts"
                / "run_ascend_benchmark_ci.sh"
            ),
        ]
        env = _build_script_env(
            layout=layout,
            manifest=manifest,
            materialized=materialized,
            result_root=result_root,
        )
        completed = subprocess.run(
            command,
            cwd=trigger_repo_path,
            check=False,
            env={**os.environ, **env},
        )
        if completed.returncode != 0:
            return completed.returncode

        raw_result_file = result_root / "raw_benchmark.json"
        server_log = result_root / "server.log"
        if not raw_result_file.is_file():
            raise ValueError(f"perf smoke run did not produce raw result: {raw_result_file}")

        perf_smoke_result = build_perf_smoke_result(
            manifest=manifest,
            materialized=materialized,
            raw_result_file=raw_result_file,
            server_log=server_log,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(perf_smoke_result, indent=2) + "\n", encoding="utf-8")
        return 0
    finally:
        for repo_key, payload in materialized.items():
            worktree_path = Path(payload["path"])
            source_repo = Path(source_paths[repo_key])
            _remove_worktree(source_repo, worktree_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-ascend-perf-smoke",
        description="Materialize a combo manifest and run the current Ascend benchmark CI script as an L1 smoke benchmark.",
    )
    parser.add_argument("--combo-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--runtime-root")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    layout = resolve_repo_layout()
    return run_ascend_perf_smoke(
        layout=layout,
        manifest_path=Path(args.combo_manifest).resolve(),
        output_path=Path(args.output).resolve(),
        result_root=Path(args.result_root).resolve(),
        runtime_root=Path(args.runtime_root).resolve() if args.runtime_root else None,
        execute=args.execute,
    )


if __name__ == "__main__":
    raise SystemExit(main())
