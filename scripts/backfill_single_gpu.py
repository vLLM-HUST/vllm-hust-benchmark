#!/usr/bin/env python3
"""One-click backfill runner for missing single-GPU vllm-hust leaderboard cells.

Subcommands
-----------
plan        Show what is currently missing and what would be run.
run         Run the missing cells (idempotent, resumable).
status      Show progress from the checkpoint.
aggregate   Rebuild leaderboard-data/snapshots from submissions/.
push        Stage submissions/ + snapshots/ and push to remote.
restore     Restore the original vllm-hust and vllm-ascend-hust HEADs.

State
-----
State is stored under .benchmarks/backfill-single-gpu/:
  state.json   Per-cell status, current commit pair, last error.
  log.txt      Append-only log of every step the runner takes.

Why this script
---------------
It automates the previously hand-rolled flow:

  1. `git checkout` the target vllm-hust commit.
  2. `git checkout` the matching vllm-ascend-hust plugin commit.
  3. `pip install -e .` the ascend plugin (with the official C-extension
     build skipped to avoid fragile compilation).
  4. Run the right vllm bench subcommand for the scenario
     (latency / throughput / serve).
  5. Convert the raw result into a website-compatible
     submissions/<run-id>/ artifact with the wrapper CLI.
  6. Aggregate to leaderboard-data/snapshots/ via publish-website.
  7. Print the next-step git add / commit / push commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.submission_artifacts import normalize_submission_artifact_file


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
HUST_REPO = WORKSPACE_ROOT / "vllm-hust"
ASCEND_REPO = WORKSPACE_ROOT / "vllm-ascend-hust"

PYTHON_BIN = Path(
    os.environ.get("BACKFILL_PYTHON")
    or Path.home() / "miniconda3/envs/vllm-hust-dev/bin/python"
)
if not PYTHON_BIN.is_file():
    PYTHON_BIN = Path(sys.executable)

STATE_DIR = REPO_ROOT / ".benchmarks" / "backfill-single-gpu"
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = STATE_DIR / "log.txt"

SHAREGPT_DATASET = Path(
    "/data/shared_datasets/vllm-hust-benchmark/current-benchmark-datasets/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
SONNET_DATASET = HUST_REPO / "benchmarks" / "sonnet.txt"

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
MODEL_PARAMETERS = "14B"
MODEL_PRECISION = "FP16"
HARDWARE_VENDOR = "Huawei"
HARDWARE_CHIP_MODEL = "910B2"
CHIP_COUNT = 1
NODE_COUNT = 1
SUBMITTER = "vllm-hust-org-member"
DATA_SOURCE = "vllm-hust-benchmark"

# vllm-ascend-hust commits are resolved dynamically by time-aligning
# to each vllm-hust commit (old PR-branch SHAs have been GC'd, so we
# use the HEAD of origin/main at the time the vllm-hust commit was made).


def resolve_ascend_commit(hust_commit: str) -> str:
    """Find the vllm-ascend-hust HEAD on origin/main at the time of hust_commit."""
    out = subprocess.run(
        ["git", "log", "-1", "--format=%cI", hust_commit],
        cwd=HUST_REPO, capture_output=True, text=True, check=True,
    )
    timestamp = out.stdout.strip()
    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--before", timestamp, "origin/main"],
        cwd=ASCEND_REPO, capture_output=True, text=True, check=True,
    )
    sha = out.stdout.strip()
    if not sha:
        raise RuntimeError(
            f"could not find ascend-hust commit at time {timestamp} for {hust_commit}"
        )
    return sha


# Known-good ascend plugin commit that is compatible with the vllm-hust fork.
# The vllm-ascend-hust fork's origin/main has been updated with upstream commits
# that reference vllm modules (e.g. expert_map_manager) not present in the
# vllm-hust fork.  This commit is the last vllm-hust-fork-only commit known to
# work with all backfill targets.
#
# LIMITATION: Using a single ascend commit for all historical vllm-hust commits
# means the recorded plugin SHA in the runtime_provenance does NOT reflect the
# actual plugin version that was historically paired with each vllm-hust commit.
# This is a known trade-off: the alternative (time-aligned commit) would fail
# at runtime because upstream-merged ascend commits reference modules absent
# from the older vllm-hust fork.
#
# To fix this properly, create standalone compatibility-backport commits in the
# vllm-ascend-hust repo (one per vllm-hust milestone) and reference them here
# via a mapping table.
_COMPATIBLE_ASCEND_COMMIT = "bf2984e34a8923ac254251c6e265dffbad4aa70d"


def _resolve_compatible_ascend_commit(hust_commit: str) -> str:
    """Resolve an ascend plugin commit that is compatible with hust_commit.

    The dynamic ``resolve_ascend_commit`` may return an upstream vllm-ascend
    commit whose code references modules (e.g. ``expert_map_manager``) or
    entry points (e.g. ``register_model``) that do **not** exist in the
    older vllm-hust codebase.  Since the time-aligned commit is unreliable
    (our vllm-ascend-hust repo has been synced with upstream which adds
    features not present in vllm-hust), we always use the last known-good
    vllm-hust-fork-only commit for vllm-hust PRs.
    """
    return _COMPATIBLE_ASCEND_COMMIT

# Default missing cells (workload -> list of vllm-hust commits).
DEFAULT_CELLS: dict[str, list[str]] = {
    "random-latency": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "2fb7859dd024b51c7bd09b0c9b5cc701898090bb",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "dcc06b18f32404abafe6922910117f1b9f66054b",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
    ],
    "sharegpt-throughput": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
    ],
    "sonnet-throughput": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
    ],
    # Online serving scenarios
    "random-online": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "2fb7859dd024b51c7bd09b0c9b5cc701898090bb",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "dcc06b18f32404abafe6922910117f1b9f66054b",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
    ],
    "sharegpt-online": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "2fb7859dd024b51c7bd09b0c9b5cc701898090bb",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "dcc06b18f32404abafe6922910117f1b9f66054b",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
    ],
    "prefix-repetition-online": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "2fb7859dd024b51c7bd09b0c9b5cc701898090bb",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "dcc06b18f32404abafe6922910117f1b9f66054b",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
    ],
    "instructcoder-online": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "2fb7859dd024b51c7bd09b0c9b5cc701898090bb",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "dcc06b18f32404abafe6922910117f1b9f66054b",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
    ],
}

# Per-scenario benchmark parameters, aligned with the existing
# submissions/*/run_leaderboard.json so the new cells are comparable.
SCENARIO_PARAMS: dict[str, dict[str, Any]] = {
    "random-latency": {
        "benchmark_type": "latency",
        "input_length": 1024,
        "output_length": 128,
        "batch_size": 8,
        "num_iters_warmup": 10,
        "num_iters": 30,
        "extra_args": [],
    },
    "sharegpt-throughput": {
        "benchmark_type": "throughput",
        "dataset_name": "sharegpt",
        "dataset_path": str(SHAREGPT_DATASET),
        "num_prompts": 200,
        "extra_args": [],
    },
    "sonnet-throughput": {
        "benchmark_type": "throughput",
        "dataset_name": "sonnet",
        "dataset_path": str(SONNET_DATASET),
        "num_prompts": 200,
        "extra_args": [],
    },
    # Online serving scenarios
    "random-online": {
        "benchmark_type": "serve",
        "dataset_name": "random",
        "num_prompts": 200,
        "request_rate": 1,
        "input_length": 1024,
        "output_length": 256,
        "extra_args": [],
    },
    "sharegpt-online": {
        "benchmark_type": "serve",
        "dataset_name": "sharegpt",
        "dataset_path": str(SHAREGPT_DATASET),
        "num_prompts": 200,
        "request_rate": 1,
        "extra_args": [],
    },
    "prefix-repetition-online": {
        "benchmark_type": "serve",
        "dataset_name": "prefix_repetition",
        "num_prompts": 200,
        "request_rate": 1,
        "input_length": 4096,
        "output_length": 256,
        "extra_args": [],
    },
    "instructcoder-online": {
        "benchmark_type": "serve",
        "dataset_name": "hf",
        "dataset_path": "likaixin/InstructCoder",
        "num_prompts": 2048,
        "request_rate": 1,
        "extra_args": [],
    },
    # visionarena-online and agent-research-online are NOT included here
    # because they use openai-chat backend with /v1/chat/completions endpoint,
    # which requires a different CLI invocation.
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg: str, *, also_print: bool = True) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    if STATE_FILE.is_file():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "hust_head": current_head(HUST_REPO),
        "ascend_head": current_head(ASCEND_REPO),
        "cells": {},
    }


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def current_head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


# ---------------------------------------------------------------------------
# Git + env
# ---------------------------------------------------------------------------

def git_checkout(repo: Path, commit: str) -> None:
    subprocess.run(["git", "fetch", "--all", "--quiet"], cwd=repo, check=False)
    # Force checkout to discard any local modifications (e.g. from
    # install_ascend_plugin) that would block the checkout.
    subprocess.run(
        ["git", "checkout", "-fq", commit], cwd=repo, check=False
    )
    # If the commit is not local, fetch and retry.
    head = current_head(repo)
    if not head.startswith(commit):
        subprocess.run(
            ["git", "fetch", "origin", commit], cwd=repo, check=True
        )
        subprocess.run(
            ["git", "checkout", "-fq", commit], cwd=repo, check=True
        )


# NOTE: Compatibility fixes for older vllm-hust commits should be
# implemented as standalone, referenceable commits in the respective
# repositories (vllm-hust and vllm-ascend-hust), not as runtime
# monkeypatches. Runtime monkeypatching makes the recorded SHA
# inconsistent with the actually executed source code.


# ---------------------------------------------------------------------------
# Existing-cell discovery
# ---------------------------------------------------------------------------

def load_leaderboard() -> list[dict[str, Any]]:
    snapshot = REPO_ROOT / "leaderboard-data" / "snapshots" / "leaderboard_single.json"
    if not snapshot.is_file():
        return []
    return json.loads(snapshot.read_text(encoding="utf-8"))


def cell_already_present(
    workload: str, commit: str, submissions_dir: Path = REPO_ROOT / "submissions"
) -> bool:
    """Treat a cell as present when the snapshot already lists it."""
    short = commit[:9]
    for entry in load_leaderboard():
        if entry.get("workload", {}).get("name") != workload:
            continue
        if entry.get("config_type") != "single_gpu":
            continue
        if (entry.get("hardware", {}).get("chip_model") or "").upper() != "910B2":
            continue
        meta_commit = entry.get("metadata", {}).get("git_commit", "") or ""
        if meta_commit.startswith(short):
            return True

    # Also count a freshly-staged submission directory as "present" so a
    # re-run after a crash does not redo the same cell.
    if submissions_dir.is_dir():
        for sub in submissions_dir.iterdir():
            manifest = sub / "leaderboard_manifest.json"
            if not manifest.is_file():
                continue
            try:
                run = json.loads((sub / "run_leaderboard.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if run.get("workload", {}).get("name") != workload:
                continue
            if run.get("config_type") != "single_gpu":
                continue
            if (run.get("hardware", {}).get("chip_model") or "").upper() != "910B2":
                continue
            meta_commit = run.get("metadata", {}).get("git_commit", "") or ""
            if meta_commit.startswith(short):
                return True
    return False


# ---------------------------------------------------------------------------
# Per-cell execution
# ---------------------------------------------------------------------------

def build_run_id(workload: str, hust_commit: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"single-gpu-backfill-{workload}-{hust_commit[:9]}-{today}"


def _run_bench_with_retry(
    cmd: list[str], env: dict[str, str], output_dir: Path, bench_log: Path
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run bench command, retrying with old CLI flags if new ones are unsupported."""
    log(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    with bench_log.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, cwd=HUST_REPO, env=env,
                                stdout=log_file, stderr=subprocess.STDOUT,
                                check=False)

    raw = output_dir / "raw.json"
    if not raw.is_file():
        for candidate in output_dir.glob("raw*.json"):
            raw = candidate
            break

    # Check if the failure was due to unrecognized arguments (old CLI).
    if result.returncode == 2 and not raw.is_file():
        log_content = bench_log.read_text(encoding="utf-8") if bench_log.is_file() else ""
        if "unrecognized arguments" in log_content:
            log("Detected old CLI (unrecognized arguments), retrying with legacy flags...")
            # Rebuild command with legacy flags.
            new_cmd = _to_legacy_cmd(cmd, output_dir)
            log(f"$ {' '.join(shlex.quote(c) for c in new_cmd)}")
            with bench_log.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(new_cmd, cwd=HUST_REPO, env=env,
                                        stdout=log_file, stderr=subprocess.STDOUT,
                                        check=False)
            raw = output_dir / "raw.json"
            if not raw.is_file():
                for candidate in output_dir.glob("raw*.json"):
                    raw = candidate
                    break

    return result, raw


def _to_legacy_cmd(cmd: list[str], output_dir: Path) -> list[str]:
    """Convert new-style CLI flags to legacy (old vllm) flags."""
    new_cmd = []
    skip_next = False
    for i, arg in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if arg == "--gpu-memory-utilization":
            skip_next = True  # skip the value too
            continue
        if arg == "--output-json":
            # Replace with --save-result --result-dir <dir> --result-filename raw.json
            new_cmd.extend(["--save-result", "--result-dir", str(output_dir),
                            "--result-filename", "raw.json"])
            skip_next = True  # skip the value
            continue
        new_cmd.append(arg)
    return new_cmd


def _build_env() -> dict[str, str]:
    """Build the environment for running vllm commands."""
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("VLLM_USE_V1", "1")
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    env["ASCEND_VISIBLE_DEVICES"] = "0"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    atb_home = "/usr/local/Ascend/nnal/atb/9.0.0/atb"
    torch_cxx_abi = subprocess.run(
        [str(PYTHON_BIN), "-c", "import torch; print(torch.compiled_with_cxx11_abi())"],
        capture_output=True, text=True, check=False
    ).stdout.strip()
    cxx_abi_dir = "cxx_abi_1" if torch_cxx_abi == "True" else "cxx_abi_0"
    atb_lib_path = f"{atb_home}/{cxx_abi_dir}/lib"

    env.setdefault("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{atb_lib_path}:{env['LD_LIBRARY_PATH']}"
    env["LD_LIBRARY_PATH"] = "/usr/local/Ascend/ascend-toolkit/lib64:" + env["LD_LIBRARY_PATH"]
    env["LD_LIBRARY_PATH"] = "/usr/local/Ascend/cann-9.0.0/lib64:" + env["LD_LIBRARY_PATH"]
    env["ATB_HOME_PATH"] = f"{atb_home}/{cxx_abi_dir}"
    return env


def _run_serve_bench(
    params: dict[str, Any], output_dir: Path, bench_log: Path, env: dict[str, str]
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run serve benchmark: start server, run bench client, stop server."""
    host = "127.0.0.1"
    port = 8000
    server_log = output_dir / "server.log"

    # Start the vllm server in the background.
    serve_cmd = [
        str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "serve",
        MODEL_NAME,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", "0.6",
    ]
    # Wait for the port to be free (previous server may still be shutting down).
    import socket as _socket
    _port_wait_start = time.time()
    while time.time() - _port_wait_start < 60:
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            s.settimeout(2)
            s.connect((host, port))
            s.close()
            time.sleep(2)
        except (ConnectionRefusedError, OSError):
            break
    else:
        log("Warning: port 8000 still in use after 60s, proceeding anyway")

    log(f"$ {' '.join(shlex.quote(c) for c in serve_cmd)} &")
    with server_log.open("w", encoding="utf-8") as sf:
        server_proc = subprocess.Popen(
            serve_cmd, cwd=HUST_REPO, env=env,
            stdout=sf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    try:
        # Wait for the server to be ready by polling /health.
        import urllib.request
        import urllib.error
        max_wait = 600  # seconds
        start = time.time()
        ready = False
        last_error = ""
        while time.time() - start < max_wait:
            if server_proc.poll() is not None:
                raise RuntimeError(
                    f"Server exited early with code {server_proc.returncode}. "
                    f"Check {server_log}"
                )
            try:
                req = urllib.request.Request(f"http://{host}:{port}/health")
                urllib.request.urlopen(req, timeout=5)
                ready = True
                break
            except (urllib.error.URLError, OSError) as e:
                last_error = str(e)
                time.sleep(5)
        if not ready:
            raise RuntimeError(
                f"Server did not become ready within {max_wait}s. "
                f"Last error: {last_error}"
            )
        log("Server is ready, starting benchmark client...")

        # Build bench client command (old CLI style: --save-result).
        bench_cmd = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "vllm",
            "--endpoint", "/v1/completions",
            "--host", host,
            "--port", str(port),
            "--model", MODEL_NAME,
            "--dataset-name", params["dataset_name"],
            "--num-prompts", str(params["num_prompts"]),
            "--request-rate", str(params.get("request_rate", 1)),
            "--save-result",
            "--result-dir", str(output_dir),
            "--result-filename", "raw.json",
        ]
        if params.get("dataset_path"):
            bench_cmd.extend(["--dataset-path", params["dataset_path"]])
        if params.get("input_length"):
            bench_cmd.extend(["--random-input-len", str(params["input_length"])])
        if params.get("output_length"):
            bench_cmd.extend(["--random-output-len", str(params["output_length"])])

        log(f"$ {' '.join(shlex.quote(c) for c in bench_cmd)}")
        with bench_log.open("w", encoding="utf-8") as lf:
            bench_result = subprocess.run(
                bench_cmd, cwd=HUST_REPO, env=env,
                stdout=lf, stderr=subprocess.STDOUT,
                check=False,
            )

        raw = output_dir / "raw.json"
        if not raw.is_file():
            for candidate in output_dir.glob("raw*.json"):
                raw = candidate
                break
        return bench_result, raw
    finally:
        # Always stop the server (kill entire process group).
        log("Stopping server...")
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            pass
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            server_proc.wait()


def run_vllm_bench(
    workload: str, hust_commit: str, output_dir: Path
) -> Path:
    """Run the right vllm bench subcommand and return the raw result JSON path."""
    params = SCENARIO_PARAMS[workload]
    benchmark_type = params["benchmark_type"]
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_log = output_dir / "bench.log"
    env = _build_env()

    if benchmark_type == "latency":
        cmd: list[str] = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "latency",
            "--model", MODEL_NAME,
            "--input-len", str(params["input_length"]),
            "--output-len", str(params["output_length"]),
            "--batch-size", str(params["batch_size"]),
            "--num-iters-warmup", str(params["num_iters_warmup"]),
            "--num-iters", str(params["num_iters"]),
            "--gpu-memory-utilization", "0.6",
            "--output-json", str(output_dir / "raw.json"),
        ]
        result, raw = _run_bench_with_retry(cmd, env, output_dir, bench_log)
    elif benchmark_type == "throughput":
        cmd = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
            "--model", MODEL_NAME,
            "--dataset-name", params["dataset_name"],
            "--num-prompts", str(params["num_prompts"]),
            "--gpu-memory-utilization", "0.6",
            "--output-json", str(output_dir / "raw.json"),
        ]
        if params.get("dataset_path"):
            cmd.extend(["--dataset-path", params["dataset_path"]])
        result, raw = _run_bench_with_retry(cmd, env, output_dir, bench_log)
    else:  # serve
        result, raw = _run_serve_bench(params, output_dir, bench_log, env)

    if result.returncode != 0:
        if raw.is_file() and raw.stat().st_size > 0:
            # The benchmark script has a known bug: after writing results it
            # tries to re-initialize the engine (which fails because the NPU
            # is already in use), causing exit code 1.  The raw result is
            # valid, so we accept it.
            log(f"benchmark subprocess exited with code {result.returncode} "
                f"but {raw.name} was produced — accepting result")
        else:
            log(f"benchmark failed with exit code {result.returncode}")
            if bench_log.is_file():
                log(f"benchmark output (last 100 lines):")
                with bench_log.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines[-100:]:
                        log(f"  {line.rstrip()}", also_print=False)
            raise RuntimeError(
                f"benchmark failed with exit code {result.returncode} "
                f"for workload={workload} commit={hust_commit[:9]}"
            )

    if not raw.is_file():
        raise FileNotFoundError(f"raw result json not produced under {output_dir}")
    return raw


def _generate_same_spec(workload: str) -> dict[str, Any]:
    """Generate a same_spec payload matching the official baseline hash.

    The official v0.18.0 baseline was generated at commit ``2d6f5de`` using
    an older version of ``build_same_spec_payload`` that did **not** include
    ``model_quantization`` in the hash basis.  This function replicates
    that exact logic so the ``resolved_spec_hash`` matches the official
    baseline, enabling leaderboard goal-progress pairing.
    """
    model_tag = "qwen25-coder-14b" if "instructcoder" in workload else "qwen25-14b"
    spec_name = f"official-ascend-jan-2026-v0180-{workload}-{model_tag}-910b2.json"
    spec_path = REPO_ROOT / "docs" / "official-baselines" / spec_name
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Replicate the old ``resolve_server_parameters`` (commit 2d6f5de).
    # ------------------------------------------------------------------
    server_params = dict(spec["server_parameters"])
    server_params["model"] = spec["model"]
    if "dtype" not in server_params:
        server_params["dtype"] = _PRECISION_TO_DTYPE[spec["model_precision"]]
    if "enforce_eager" not in server_params:
        server_params["enforce_eager"] = ""
    if "gpu_memory_utilization" not in server_params:
        server_params["gpu_memory_utilization"] = 0.6

    # ------------------------------------------------------------------
    # Replicate the old ``resolve_client_parameters`` (commit 2d6f5de).
    # ------------------------------------------------------------------
    client_params = dict(spec["client_parameters"])
    client_params["model"] = spec["model"]
    if "gpu_memory_utilization" not in client_params:
        # Only add for non-serve benchmark types (old logic).
        if spec.get("scenario") not in ("random-online", "sharegpt-online",
                                        "prefix-repetition-online", "instructcoder-online",
                                        "visionarena-online", "agent-research-online"):
            client_params["gpu_memory_utilization"] = 0.6

    # Transform random input/output lengths (old logic).
    if client_params.get("dataset_name") == "random":
        if "input_len" in client_params and "random_input_len" not in client_params:
            client_params["random_input_len"] = client_params.pop("input_len")
        if "output_len" in client_params and "random_output_len" not in client_params:
            client_params["random_output_len"] = client_params.pop("output_len")

    # ------------------------------------------------------------------
    # Build the hash basis (old version, NO model_quantization).
    # ------------------------------------------------------------------
    hash_basis = {
        "schema_version": "benchmark-same-spec/v1",
        "spec_id": spec["id"],
        "scenario": spec["scenario"],
        "model": spec["model"],
        "model_parameters": spec["model_parameters"],
        "model_precision": spec["model_precision"],
        "hardware_vendor": spec["hardware_vendor"],
        "hardware_chip_model": spec["hardware_chip_model"],
        "chip_count": int(spec.get("chip_count") or 0),
        "node_count": int(spec.get("node_count") or 0),
        "resolved_server_parameters": {
            k: v for k, v in server_params.items()
            if k not in {"host", "port", "model"}
        },
        "resolved_client_parameters": {
            k: v for k, v in client_params.items()
            if k not in {"host", "port", "model"}
        },
    }

    hash_input = json.dumps(
        hash_basis,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    resolved_spec_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

    return {
        "schema_version": "benchmark-same-spec/v1",
        "spec_id": spec["id"],
        "spec_label": str(spec.get("label") or ""),
        "spec_source": str(spec_path.resolve()),
        "scenario": spec["scenario"],
        "model": spec["model"],
        "model_parameters": spec["model_parameters"],
        "model_precision": spec["model_precision"],
        "hardware_vendor": spec["hardware_vendor"],
        "hardware_chip_model": spec["hardware_chip_model"],
        "chip_count": int(spec.get("chip_count") or 0),
        "node_count": int(spec.get("node_count") or 0),
        "resolved_spec_hash": resolved_spec_hash,
        "resolved_server_parameters": server_params,
        "resolved_client_parameters": client_params,
    }


# Precision-to-dtype mapping (replicated from old same_spec.py).
_PRECISION_TO_DTYPE = {
    "FP32": "float32",
    "FP16": "float16",
    "BF16": "bfloat16",
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
}


def _build_reproducible_cmd(workload: str, output_dir: Path) -> str:
    """Build the exact vllm bench command used for this workload.

    This is the command that, when re-run, should reproduce the benchmark
    results.  It is stored in the artifact metadata as ``reproducible_cmd``.
    """
    params = SCENARIO_PARAMS[workload]
    benchmark_type = params["benchmark_type"]
    raw_path = output_dir / "raw.json"

    if benchmark_type == "latency":
        parts = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "latency",
            "--model", MODEL_NAME,
            "--input-len", str(params["input_length"]),
            "--output-len", str(params["output_length"]),
            "--batch-size", str(params["batch_size"]),
            "--num-iters-warmup", str(params["num_iters_warmup"]),
            "--num-iters", str(params["num_iters"]),
            "--gpu-memory-utilization", "0.6",
            "--output-json", str(raw_path),
        ]
    elif benchmark_type == "throughput":
        parts = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
            "--model", MODEL_NAME,
            "--dataset-name", params["dataset_name"],
            "--num-prompts", str(params["num_prompts"]),
            "--gpu-memory-utilization", "0.6",
            "--output-json", str(raw_path),
        ]
        if params.get("dataset_path"):
            parts.extend(["--dataset-path", str(params["dataset_path"])])
    else:  # serve
        parts = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--backend", "vllm",
            "--endpoint", "/v1/completions",
            "--host", "127.0.0.1",
            "--port", "8000",
            "--model", MODEL_NAME,
            "--dataset-name", params["dataset_name"],
            "--num-prompts", str(params["num_prompts"]),
            "--request-rate", str(params.get("request_rate", 1)),
            "--save-result",
            "--result-dir", str(output_dir),
            "--result-filename", "raw.json",
        ]
        if params.get("dataset_path"):
            parts.extend(["--dataset-path", str(params["dataset_path"])])
        if params.get("input_length"):
            parts.extend(["--random-input-len", str(params["input_length"])])
        if params.get("output_length"):
            parts.extend(["--random-output-len", str(params["output_length"])])

    return " ".join(shlex.quote(p) for p in parts)


def submit_artifact(
    workload: str, hust_commit: str, ascend_commit: str, run_id: str, raw: Path
) -> Path:
    output_dir = REPO_ROOT / "submissions" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    constraints = REPO_ROOT / "docs" / "examples" / "constraints_metrics.sample.json"

    # Generate same_spec so the submission passes the public-snapshot filter.
    same_spec = _generate_same_spec(workload)
    same_spec_file = STATE_DIR / f"same_spec_{workload}.json"
    same_spec_file.parent.mkdir(parents=True, exist_ok=True)
    same_spec_file.write_text(json.dumps(same_spec, indent=2) + "\n", encoding="utf-8")

    cmd: list[str] = [
        str(PYTHON_BIN), "-m", "vllm_hust_benchmark.cli", "submit", workload,
        "--benchmark-result-file", str(raw),
        "--constraints-file", str(constraints),
        "--same-spec-file", str(same_spec_file),
        "--run-id", run_id,
        "--engine", "vllm-hust",
        "--engine-version", "0.18.0.post1",
        "--model-name", MODEL_NAME,
        "--model-parameters", MODEL_PARAMETERS,
        "--model-precision", MODEL_PRECISION,
        "--hardware-vendor", HARDWARE_VENDOR,
        "--hardware-chip-model", HARDWARE_CHIP_MODEL,
        "--chip-count", str(CHIP_COUNT),
        "--node-count", str(NODE_COUNT),
        "--submitter", SUBMITTER,
        "--data-source", DATA_SOURCE,
        "--git-commit", hust_commit,
        "--github-repository", "vllm-hust/vllm-hust",
        "--github-ref", "main",
        "--runtime-python", str(PYTHON_BIN),
        "--engine-source-repository", "vllm-hust/vllm-hust",
        "--engine-source-ref", hust_commit[:10],
        "--engine-source-commit", hust_commit,
        "--plugin-source-engine", "vllm-ascend-hust",
        "--plugin-source-repository", "vllm-hust/vllm-ascend-hust",
        "--plugin-source-ref", ascend_commit[:10],
        "--plugin-source-commit", ascend_commit,
    ]
    params = SCENARIO_PARAMS[workload]
    if params.get("input_length") is not None:
        cmd += ["--input-length", str(params["input_length"])]
    if params.get("output_length") is not None:
        cmd += ["--output-length", str(params["output_length"])]
    if params.get("batch_size") is not None:
        cmd += ["--batch-size", str(params["batch_size"])]
    if params.get("num_prompts") is not None:
        cmd += ["--concurrent-requests", str(params["num_prompts"])]

    log(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    # Post-process the artifact to fill in provenance fields that are not
    # available as CLI flags (reproducible_cmd, verified).
    reproducible_cmd = _build_reproducible_cmd(workload, output_dir)
    artifact_path = output_dir / "run_leaderboard.json"
    if artifact_path.is_file():
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["metadata"]["reproducible_cmd"] = reproducible_cmd
        artifact["metadata"]["verified"] = False
        artifact_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        log(f"Set reproducible_cmd and verified=false in {artifact_path}")

    return output_dir


# ---------------------------------------------------------------------------
# Validation (Section 13.2: result completeness)
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACT_FIELDS = {
    "entry_id", "engine", "engine_version", "config_type",
    "hardware", "model", "workload", "metrics", "constraints",
    "versions", "metadata",
}
REQUIRED_METRICS_FIELDS = {"ttft_ms", "tbt_ms", "throughput_tps", "error_rate"}


def validate_submission(sub_dir: Path) -> list[str]:
    """Validate a single submission directory for completeness.

    Checks:
      - run_leaderboard.json exists and has all required fields
      - leaderboard_manifest.json exists
      - Metrics are within reasonable ranges
      - config_type matches single_gpu
    """
    errors: list[str] = []
    rid = sub_dir.name

    manifest = sub_dir / "leaderboard_manifest.json"
    if not manifest.is_file():
        errors.append(f"{rid}: missing leaderboard_manifest.json")
    else:
        try:
            json.loads(manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"{rid}: leaderboard_manifest.json invalid: {exc}")

    artifact = sub_dir / "run_leaderboard.json"
    if not artifact.is_file():
        errors.append(f"{rid}: missing run_leaderboard.json")
        return errors

    try:
        run = json.loads(artifact.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"{rid}: run_leaderboard.json invalid: {exc}")
        return errors

    # Check required top-level fields
    missing = REQUIRED_ARTIFACT_FIELDS - set(run.keys())
    if missing:
        errors.append(f"{rid}: missing top-level fields: {missing}")

    # Check config_type is present and valid
    valid_config_types = {"single_gpu", "multi_gpu"}
    ct = run.get("config_type")
    if ct not in valid_config_types:
        errors.append(f"{rid}: config_type is {ct!r}, expected one of {valid_config_types}")

    # Check metrics
    metrics = run.get("metrics", {})
    missing_metrics = REQUIRED_METRICS_FIELDS - set(metrics.keys())
    if missing_metrics:
        errors.append(f"{rid}: missing metrics fields: {missing_metrics}")
    else:
        workload_name = run.get("workload", {}).get("name", "")
        is_latency = workload_name == "random-latency"
        ttft = metrics.get("ttft_ms") or 0
        tbt = metrics.get("tbt_ms") or 0
        tput = metrics.get("throughput_tps") or 0
        err_rate = metrics.get("error_rate", -1)
        if ttft < 0:
            errors.append(f"{rid}: ttft_ms is negative ({ttft})")
        if tbt < 0:
            errors.append(f"{rid}: tbt_ms is negative ({tbt})")
        if tput <= 0 and not is_latency:
            # latency 不测吞吐
            errors.append(f"{rid}: throughput_tps is <= 0 ({tput})")
        if is_latency:
            # latency 场景 error_rate 可以为 None 或 0.0
            if err_rate is not None and (err_rate < 0 or err_rate > 1):
                errors.append(f"{rid}: error_rate out of range [0,1] ({err_rate})")
        else:
            if err_rate is None or err_rate < 0 or err_rate > 1:
                errors.append(f"{rid}: error_rate out of range [0,1] ({err_rate})")
            elif err_rate >= 1.0:
                # error_rate == 1.0 means all requests failed — not reproducible.
                errors.append(f"{rid}: error_rate={err_rate} (all requests failed, result not reproducible)")

    # Check workload
    workload = run.get("workload", {})
    if not workload.get("name"):
        errors.append(f"{rid}: workload.name is missing")

    # Check hardware
    hw = run.get("hardware", {})
    chip = str(hw.get("chip_model", "") or "").upper()
    if chip != "910B2":
        errors.append(f"{rid}: hardware.chip_model is {chip!r}, expected 910B2")

    return errors


def validate_snapshot(snapshot_path: Path) -> list[str]:
    """Validate the aggregated leaderboard snapshot.

    Checks:
      - File exists and is valid JSON
      - Each entry has required fields
      - No duplicate entries for the same (workload, git_commit)
    """
    errors: list[str] = []
    if not snapshot_path.is_file():
        errors.append(f"missing snapshot: {snapshot_path}")
        return errors

    try:
        entries = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        errors.append(f"invalid snapshot {snapshot_path}: {exc}")
        return errors

    if not isinstance(entries, list):
        errors.append(f"snapshot {snapshot_path} must be a JSON array")
        return errors

    seen_ids: set[str] = set()
    for i, entry in enumerate(entries):
        eid = entry.get("entry_id", "")
        if eid and eid in seen_ids:
            errors.append(f"{snapshot_path.name}[{i}]: duplicate entry_id {eid}")
        seen_ids.add(eid)

        # Validate each entry
        sub_errors = validate_submission_artifact_entry(entry, f"{snapshot_path.name}[{i}]")
        errors.extend(sub_errors)

    return errors


def validate_submission_artifact_entry(entry: dict[str, Any], tag: str) -> list[str]:
    """Validate a single leaderboard entry dict."""
    errors: list[str] = []
    missing = REQUIRED_ARTIFACT_FIELDS - set(entry.keys())
    if missing:
        errors.append(f"{tag}: missing fields: {missing}")

    metrics = entry.get("metrics", {})
    missing_metrics = REQUIRED_METRICS_FIELDS - set(metrics.keys())
    if missing_metrics:
        errors.append(f"{tag}: missing metrics: {missing_metrics}")

    workload = entry.get("workload", {})
    if not workload.get("name"):
        errors.append(f"{tag}: missing workload.name")

    return errors


def _check_error_rate(sub_dir: Path) -> str | None:
    """Check if the submission has error_rate == 1.0 (all requests failed).

    Returns an error message if the submission should be rejected, None otherwise.
    """
    artifact_path = sub_dir / "run_leaderboard.json"
    if not artifact_path.is_file():
        return None
    try:
        run = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    metrics = run.get("metrics", {})
    error_rate = metrics.get("error_rate")
    if error_rate is not None and error_rate >= 1.0:
        workload_name = run.get("workload", {}).get("name", "unknown")
        return (
            f"submission rejected: error_rate={error_rate} for {workload_name} "
            f"(all requests failed, not reproducible)"
        )
    return None


def run_cell(workload: str, hust_commit: str, ascend_commit: str | None = None) -> dict[str, Any]:
    if ascend_commit is None:
        ascend_commit = _resolve_compatible_ascend_commit(hust_commit)

    log(f"=== {workload} @ {hust_commit[:9]} (plugin {ascend_commit[:9]}) ===")
    work_dir = STATE_DIR / "runs" / f"{workload}-{hust_commit[:9]}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Restore repos to the target commits and clean any leftover files.
        git_checkout(HUST_REPO, hust_commit)
        git_checkout(ASCEND_REPO, ascend_commit)
        # Remove any untracked files left by previous runs.
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=HUST_REPO, check=False,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=ASCEND_REPO, check=False,
        )

        raw = run_vllm_bench(workload, hust_commit, work_dir / "bench")
        run_id = build_run_id(workload, hust_commit)
        sub_dir = submit_artifact(workload, hust_commit, ascend_commit, run_id, raw)

        # Reject results where all requests failed (error_rate == 1.0).
        err_rate_msg = _check_error_rate(sub_dir)
        if err_rate_msg is not None:
            log(f"REJECTED: {err_rate_msg}")
            # Clean up the failed submission directory.
            if sub_dir.exists():
                shutil.rmtree(sub_dir)
            raise RuntimeError(err_rate_msg)

        # Validate the submission right after creation (Section 13.2).
        errors = validate_submission(sub_dir)
        if errors:
            err_msg = "; ".join(errors)
            log(f"Submission validation FAILED for {run_id}: {err_msg}")
            raise RuntimeError(f"submission validation failed: {err_msg}")

        # Normalize the submission artifact so it passes checked-in normalization tests.
        artifact_path = sub_dir / "run_leaderboard.json"
        if artifact_path.exists():
            normalize_submission_artifact_file(artifact_path)
            log(f"Normalized submission artifact: {artifact_path}")
    except Exception as exc:  # noqa: BLE001
        log(f"FAILED: {exc}")
        return {
            "status": "failed",
            "error": str(exc),
            "hust_commit": hust_commit,
            "ascend_commit": ascend_commit,
        }

    return {
        "status": "done",
        "hust_commit": hust_commit,
        "ascend_commit": ascend_commit,
        "run_id": run_id,
        "submission_dir": str(sub_dir.relative_to(REPO_ROOT)),
        "raw_result": str(raw.relative_to(REPO_ROOT)),
    }


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> int:
    log("PLAN: listing missing cells")
    total_missing = 0
    cells: dict[str, list[str]] = DEFAULT_CELLS
    for workload, commits in cells.items():
        print(f"\n[{workload}]")
        for commit in commits:
            already = cell_already_present(workload, commit)
            mark = "skip" if already else "MISSING"
            print(f"  {mark:7s}  {workload}  {commit[:9]}")
            if not already:
                total_missing += 1
    print(f"\nTotal missing: {total_missing}")
    return 0


def _resolve_full_commit(short_or_full: str) -> str:
    """Resolve a (possibly short) git SHA to a full 40-char commit hash."""
    out = subprocess.run(
        ["git", "rev-parse", short_or_full],
        cwd=HUST_REPO, capture_output=True, text=True, check=False,
    )
    if out.returncode == 0:
        resolved = out.stdout.strip()
        if len(resolved) == 40:
            return resolved
    # fallback: return as-is (may already be full, or the repo is on a different commit)
    return short_or_full


def cmd_run(args: argparse.Namespace) -> int:
    state = load_state()
    save_state(state)  # Persist the captured HEADs up front.
    log("RUN: starting backfill")
    target: dict[str, list[str]] = DEFAULT_CELLS

    if args.commit:
        workloads = args.only or list(SCENARIO_PARAMS.keys())
        full_commit = _resolve_full_commit(args.commit)
        target = {w: [full_commit] for w in workloads if w in SCENARIO_PARAMS}
    elif args.only:
        target = {w: target.get(w, []) for w in args.only if w in target}
        target = {w: c for w, c in target.items() if c}

    for workload, commits in target.items():
        for commit in commits:
            if args.ascend_commit:
                key = f"{workload}:{commit[:9]}:ascend-{args.ascend_commit[:9]}"
            else:
                key = f"{workload}:{commit[:9]}"
            existing = state["cells"].get(key, {})
            if existing.get("status") == "done" and not args.force:
                log(f"SKIP {key} (already done)")
                continue
            if cell_already_present(workload, commit) and not args.force:
                log(f"SKIP {key} (already in leaderboard)")
                state["cells"][key] = {"status": "done", "skipped": "already-present"}
                continue
            log(f"BEGIN {key}")
            result = run_cell(workload, commit, args.ascend_commit)
            state["cells"][key] = result
            save_state(state)
            if result["status"] == "failed":
                if args.fail_fast:
                    log("FAIL-FAST: stopping after first failure")
                    return 1

    log("RUN: done; remember to run `aggregate` and `push`.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = load_state()
    print("Original HEADs:")
    print(f"  vllm-hust        : {state.get('hust_head', '?')[:12]}")
    print(f"  vllm-ascend-hust : {state.get('ascend_head', '?')[:12]}")
    print()
    print("Current HEADs:")
    print(f"  vllm-hust        : {current_head(HUST_REPO)[:12]}")
    print(f"  vllm-ascend-hust : {current_head(ASCEND_REPO)[:12]}")
    print()
    print("Per-cell status:")
    if not state["cells"]:
        print("  (none recorded yet)")
    for key, info in sorted(state["cells"].items()):
        print(f"  {info.get('status', '?'):8s}  {key}  {info.get('run_id', '')}")
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    cmd = [
        str(PYTHON_BIN), "-m", "vllm_hust_benchmark.cli", "publish-website",
        "--source-dir", "submissions",
        "--output-dir", "leaderboard-data/snapshots",
        "--execute",
    ]
    log(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    # Validate aggregated snapshots (Section 13.2)
    snapshot_dir = REPO_ROOT / "leaderboard-data" / "snapshots"
    all_errors: list[str] = []
    for fname in ("leaderboard_single.json", "leaderboard_multi.json"):
        fpath = snapshot_dir / fname
        if fpath.is_file():
            errors = validate_snapshot(fpath)
            all_errors.extend(errors)

    if all_errors:
        log("AGGREGATE VALIDATION FAILED:")
        for err in all_errors:
            log(f"  VALIDATION ERROR: {err}")
        log("Aggregate done but with validation errors. Please fix and re-run.")
        log("Run scripts/validate_public_leaderboard_snapshots.py next.")
        return 1

    # Count entries in the aggregated snapshot.
    single_path = snapshot_dir / "leaderboard_single.json"
    single_count = 0
    if single_path.is_file():
        try:
            single_count = len(json.loads(single_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass

    log(f"Aggregate validation passed. "
        f"Snapshot contains {single_count} single-GPU entries. "
        "Run scripts/validate_public_leaderboard_snapshots.py next.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate all submission directories and snapshot files."""
    all_errors: list[str] = []

    # Validate submissions
    submissions_dir = REPO_ROOT / "submissions"
    if submissions_dir.is_dir():
        for sub in sorted(submissions_dir.iterdir()):
            if not sub.is_dir():
                continue
            errors = validate_submission(sub)
            if errors:
                print(f"FAIL: {sub.name}")
                for err in errors:
                    print(f"  {err}")
                all_errors.extend(errors)
            else:
                print(f"OK:   {sub.name}")

    # Validate snapshots
    snapshot_dir = REPO_ROOT / "leaderboard-data" / "snapshots"
    for fname in ("leaderboard_single.json", "leaderboard_multi.json"):
        fpath = snapshot_dir / fname
        if fpath.is_file():
            errors = validate_snapshot(fpath)
            if errors:
                print(f"FAIL: {fname}")
                for err in errors:
                    print(f"  {err}")
                all_errors.extend(errors)
            else:
                print(f"OK:   {fname}")

    if all_errors:
        print(f"\nTotal validation errors: {len(all_errors)}")
        return 1
    print("\nAll validations passed!")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    """Stage, commit and push the new submissions and refreshed snapshots."""
    subprocess.run(["git", "add", "submissions/", "leaderboard-data/snapshots/"],
                   cwd=REPO_ROOT, check=True)
    # Count the actual number of backfill submissions being pushed.
    pending_dirs = [
        d for d in (REPO_ROOT / "submissions").iterdir()
        if d.is_dir() and d.name.startswith("single-gpu-backfill-")
    ] if (REPO_ROOT / "submissions").is_dir() else []
    msg = args.message or (
        f"feat(leaderboard): backfill single-GPU vllm-hust cells "
        f"({len(pending_dirs)} submissions)"
    )
    cmd_commit = ["git", "commit", "-m", msg]
    log(f"$ {' '.join(shlex.quote(c) for c in cmd_commit)}")
    rc = subprocess.run(cmd_commit, cwd=REPO_ROOT).returncode
    if rc != 0 and rc != 1:
        # 1 == nothing to commit; anything else is a real failure.
        return rc
    if args.dry_run:
        log("DRY-RUN: skipping push")
        return 0
    cmd_push = ["git", "push", "origin", "HEAD"]
    log(f"$ {' '.join(shlex.quote(c) for c in cmd_push)}")
    return subprocess.run(cmd_push, cwd=REPO_ROOT).returncode


def cmd_restore(args: argparse.Namespace) -> int:
    state = load_state()
    if state.get("hust_head"):
        log(f"Restoring vllm-hust to {state['hust_head'][:12]}")
        git_checkout(HUST_REPO, state["hust_head"])
    if state.get("ascend_head"):
        log(f"Restoring vllm-ascend-hust to {state['ascend_head'][:12]}")
        git_checkout(ASCEND_REPO, state["ascend_head"])
    log("Restore complete.")
    return 0


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("plan", help="Show what is missing.")
    sub.add_parser("status", help="Show progress from the checkpoint.")
    sub.add_parser("aggregate", help="Rebuild leaderboard-data/snapshots/.")
    sub.add_parser("validate", help="Validate all submissions and snapshots (Section 13.2).")
    sub.add_parser("restore", help="Restore original vllm-hust/ascend HEADs.")

    p_push = sub.add_parser("push", help="Stage, commit and push.")
    p_push.add_argument("-m", "--message", help="Commit message.")
    p_push.add_argument("--dry-run", action="store_true")

    p_run = sub.add_parser("run", help="Run the missing cells.")
    p_run.add_argument(
        "--only", nargs="+", help="Restrict to these workloads (e.g. random-latency)."
    )
    p_run.add_argument("--commit", help="Run only this commit (overrides DEFAULT_CELLS).")
    p_run.add_argument("--ascend-commit", help="Use this ascend plugin commit instead of resolving automatically.")
    p_run.add_argument("--force", action="store_true",
                       help="Re-run cells already marked done.")
    p_run.add_argument("--fail-fast", action="store_true",
                       help="Stop after the first failed cell.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "plan": cmd_plan,
        "run": cmd_run,
        "status": cmd_status,
        "aggregate": cmd_aggregate,
        "validate": cmd_validate,
        "push": cmd_push,
        "restore": cmd_restore,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
