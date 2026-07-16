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
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
    try:
        subprocess.run(
            ["git", "checkout", "-q", commit], cwd=repo, check=True
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            ["git", "fetch", "origin", commit], cwd=repo, check=True
        )
        subprocess.run(
            ["git", "checkout", "-q", commit], cwd=repo, check=True
        )


def install_ascend_plugin() -> None:
    """Fix naming conflict: rename openai.py -> openai_cmd.py.

    vllm.entrypoints.cli.openai shadows the external 'openai' PyPI package,
    causing circular imports when other vllm modules (e.g. mcp/tool.py) do
    ``from openai import ...``.  Renaming the file to openai_cmd.py and
    updating the import in main.py breaks the conflict.
    """
    cli_dir = HUST_REPO / "vllm" / "entrypoints" / "cli"
    openai_py = cli_dir / "openai.py"
    openai_cmd_py = cli_dir / "openai_cmd.py"

    # If both exist, remove openai.py (openai_cmd.py is already the renamed copy).
    if openai_py.is_file() and openai_cmd_py.is_file():
        openai_py.unlink()
        log(f"Patched: removed duplicate {openai_py}, keeping {openai_cmd_py}")
        # Fall through to update main.py if needed.

    if openai_cmd_py.is_file() and not openai_py.is_file():
        # Already renamed, just ensure main.py is up to date.
        pass
    elif openai_py.is_file() and not openai_cmd_py.is_file():
        openai_py.rename(openai_cmd_py)
        log(f"Patched: renamed {openai_py} -> {openai_cmd_py}")

    # Update all references in main.py.
    main_py = cli_dir / "main.py"
    if main_py.is_file():
        content = main_py.read_text(encoding="utf-8")
        orig = content
        content = content.replace("import vllm.entrypoints.cli.openai\n",
                                  "import vllm.entrypoints.cli.openai_cmd\n")
        content = content.replace("vllm.entrypoints.cli.openai,",
                                  "vllm.entrypoints.cli.openai_cmd,")
        if content != orig:
            main_py.write_text(content, encoding="utf-8")
            log(f"Patched: updated imports in {main_py}")
        else:
            log("Patched: imports already correct, skipping")


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


def run_vllm_bench(
    workload: str, hust_commit: str, output_dir: Path
) -> Path:
    """Run the right vllm bench subcommand and return the raw result JSON path."""
    params = SCENARIO_PARAMS[workload]
    benchmark_type = params["benchmark_type"]
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_log = output_dir / "bench.log"

    if benchmark_type == "latency":
        cmd: list[str] = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "latency",
            "--model", MODEL_NAME,
            "--input-len", str(params["input_length"]),
            "--output-len", str(params["output_length"]),
            "--batch-size", str(params["batch_size"]),
            "--num-iters-warmup", str(params["num_iters_warmup"]),
            "--num-iters", str(params["num_iters"]),
            "--gpu-memory-utilization", "0.85",
            "--output-json", str(output_dir / "raw.json"),
        ]
    elif benchmark_type == "throughput":
        cmd = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
            "--model", MODEL_NAME,
            "--dataset-name", params["dataset_name"],
            "--num-prompts", str(params["num_prompts"]),
            "--gpu-memory-utilization", "0.85",
            "--output-json", str(output_dir / "raw.json"),
        ]
        if params.get("dataset_path"):
            cmd.extend(["--dataset-path", params["dataset_path"]])
    else:  # serve
        cmd = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "serve",
            "--model", MODEL_NAME,
            "--backend", "vllm",
            "--endpoint", "/v1/completions",
            "--dataset-name", params["dataset_name"],
            "--num-prompts", str(params["num_prompts"]),
            "--gpu-memory-utilization", "0.85",
            "--output-json", str(output_dir / "raw.json"),
        ]
        if params.get("dataset_path"):
            cmd.extend(["--dataset-path", params["dataset_path"]])

    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
    env.setdefault("ASCEND_VISIBLE_DEVICES", "1")
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
            raise subprocess.CalledProcessError(result.returncode, cmd)

    if not raw.is_file():
        raise FileNotFoundError(f"raw result json not produced under {output_dir}")
    return raw


def submit_artifact(
    workload: str, hust_commit: str, ascend_commit: str, run_id: str, raw: Path
) -> Path:
    output_dir = REPO_ROOT / "submissions" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    constraints = REPO_ROOT / "docs" / "examples" / "constraints_metrics.sample.json"
    cmd: list[str] = [
        str(PYTHON_BIN), "-m", "vllm_hust_benchmark.cli", "submit", workload,
        "--benchmark-result-file", str(raw),
        "--constraints-file", str(constraints),
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
        "--github-ref", hust_commit[:10],
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
    return output_dir


def run_cell(workload: str, hust_commit: str) -> dict[str, Any]:
    ascend_commit = resolve_ascend_commit(hust_commit)

    log(f"=== {workload} @ {hust_commit[:9]} (plugin {ascend_commit[:9]}) ===")
    work_dir = STATE_DIR / "runs" / f"{workload}-{hust_commit[:9]}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Clean up any previously patched files before checkout to avoid duplicates.
        for f in HUST_REPO.glob("vllm/entrypoints/cli/openai_cmd*.py"):
            if f.name != "openai.py":
                f.unlink()
                log(f"Cleaned: removed stale {f.name}")

        git_checkout(HUST_REPO, hust_commit)
        git_checkout(ASCEND_REPO, ascend_commit)
        install_ascend_plugin()
        raw = run_vllm_bench(workload, hust_commit, work_dir / "bench")
        run_id = build_run_id(workload, hust_commit)
        sub_dir = submit_artifact(workload, hust_commit, ascend_commit, run_id, raw)
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


def cmd_run(args: argparse.Namespace) -> int:
    state = load_state()
    save_state(state)  # Persist the captured HEADs up front.
    log("RUN: starting backfill")
    target: dict[str, list[str]] = DEFAULT_CELLS

    if args.commit:
        workloads = args.only or list(SCENARIO_PARAMS.keys())
        target = {w: [args.commit] for w in workloads if w in SCENARIO_PARAMS}
    elif args.only:
        target = {w: target.get(w, []) for w in args.only if w in target}
        target = {w: c for w, c in target.items() if c}

    for workload, commits in target.items():
        for commit in commits:
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
            result = run_cell(workload, commit)
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
    log("Aggregate done. Run scripts/validate_public_leaderboard_snapshots.py next.")
    return 0


def cmd_push(args: argparse.Namespace) -> int:
    """Stage, commit and push the new submissions and refreshed snapshots."""
    subprocess.run(["git", "add", "submissions/", "leaderboard-data/snapshots/"],
                   cwd=REPO_ROOT, check=True)
    msg = args.message or "feat(leaderboard): backfill single-GPU vllm-hust cells"
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
    sub.add_parser("restore", help="Restore original vllm-hust/ascend HEADs.")

    p_push = sub.add_parser("push", help="Stage, commit and push.")
    p_push.add_argument("-m", "--message", help="Commit message.")
    p_push.add_argument("--dry-run", action="store_true")

    p_run = sub.add_parser("run", help="Run the missing cells.")
    p_run.add_argument(
        "--only", nargs="+", help="Restrict to these workloads (e.g. random-latency)."
    )
    p_run.add_argument("--commit", help="Run only this commit (overrides DEFAULT_CELLS).")
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
        "push": cmd_push,
        "restore": cmd_restore,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
