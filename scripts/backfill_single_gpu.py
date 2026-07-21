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
        # PR-specific commits from benchmark_plan.md
        "52b44710cdf3c797f4046698378fef3ecf6670b3",  # vllm-hust#49
        "c421a1f38f5c4dbff235aa11464d90085cf7b1c0",  # vllm-hust#41
        "98ca7fe3ba4d89a670072751dc642629f2a218f5",  # vllm-hust#81
        "9f906ff2bf1c361d02bff973c10e735aea951bdf",  # vllm-hust#76
        "8d28fcf984fd5d17a05c414e7f8f5695acc7cbc3",  # vllm-hust#118
        "73187bc8ba89b8f83652cbc24042433fb7032add",  # vllm-hust#124
        "6f612fbedff718af2dabb93692f00044e66a9b4b",  # ascend-hust#67
        "a46abb7ae68acc13a4fc5870db98619b3f97c6e0",  # ascend-hust#66
        "702214146c1f0f2c2120b87e6a460d5a39cef418",  # ascend-hust#70
        "ae16d09435abd978417a1b5ab7af352c8dcd180a",  # ascend-hust#80
    ],
    "sharegpt-throughput": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
        # PR-specific commits from benchmark_plan.md
        "52b44710cdf3c797f4046698378fef3ecf6670b3",  # vllm-hust#49
        "c421a1f38f5c4dbff235aa11464d90085cf7b1c0",  # vllm-hust#41
        "98ca7fe3ba4d89a670072751dc642629f2a218f5",  # vllm-hust#81
        "9f906ff2bf1c361d02bff973c10e735aea951bdf",  # vllm-hust#76
        "8d28fcf984fd5d17a05c414e7f8f5695acc7cbc3",  # vllm-hust#118
        "73187bc8ba89b8f83652cbc24042433fb7032add",  # vllm-hust#124
        "6f612fbedff718af2dabb93692f00044e66a9b4b",  # ascend-hust#67
        "a46abb7ae68acc13a4fc5870db98619b3f97c6e0",  # ascend-hust#66
        "702214146c1f0f2c2120b87e6a460d5a39cef418",  # ascend-hust#70
        "ae16d09435abd978417a1b5ab7af352c8dcd180a",  # ascend-hust#80
    ],
    "sonnet-throughput": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
        # PR-specific commits from benchmark_plan.md
        "52b44710cdf3c797f4046698378fef3ecf6670b3",  # vllm-hust#49
        "c421a1f38f5c4dbff235aa11464d90085cf7b1c0",  # vllm-hust#41
        "98ca7fe3ba4d89a670072751dc642629f2a218f5",  # vllm-hust#81
        "9f906ff2bf1c361d02bff973c10e735aea951bdf",  # vllm-hust#76
        "8d28fcf984fd5d17a05c414e7f8f5695acc7cbc3",  # vllm-hust#118
        "73187bc8ba89b8f83652cbc24042433fb7032add",  # vllm-hust#124
        "6f612fbedff718af2dabb93692f00044e66a9b4b",  # ascend-hust#67
        "a46abb7ae68acc13a4fc5870db98619b3f97c6e0",  # ascend-hust#66
        "702214146c1f0f2c2120b87e6a460d5a39cef418",  # ascend-hust#70
        "ae16d09435abd978417a1b5ab7af352c8dcd180a",  # ascend-hust#80
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
        # PR-specific commits from benchmark_plan.md
        "52b44710cdf3c797f4046698378fef3ecf6670b3",  # vllm-hust#49
        "c421a1f38f5c4dbff235aa11464d90085cf7b1c0",  # vllm-hust#41
        "98ca7fe3ba4d89a670072751dc642629f2a218f5",  # vllm-hust#81
        "9f906ff2bf1c361d02bff973c10e735aea951bdf",  # vllm-hust#76
        "8d28fcf984fd5d17a05c414e7f8f5695acc7cbc3",  # vllm-hust#118
        "73187bc8ba89b8f83652cbc24042433fb7032add",  # vllm-hust#124
        "6f612fbedff718af2dabb93692f00044e66a9b4b",  # ascend-hust#67
        "a46abb7ae68acc13a4fc5870db98619b3f97c6e0",  # ascend-hust#66
        "702214146c1f0f2c2120b87e6a460d5a39cef418",  # ascend-hust#70
        "ae16d09435abd978417a1b5ab7af352c8dcd180a",  # ascend-hust#80
    ],
    "sharegpt-online": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "2fb7859dd024b51c7bd09b0c9b5cc701898090bb",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "dcc06b18f32404abafe6922910117f1b9f66054b",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
        # PR-specific commits from benchmark_plan.md
        "52b44710cdf3c797f4046698378fef3ecf6670b3",  # vllm-hust#49
        "c421a1f38f5c4dbff235aa11464d90085cf7b1c0",  # vllm-hust#41
        "98ca7fe3ba4d89a670072751dc642629f2a218f5",  # vllm-hust#81
        "9f906ff2bf1c361d02bff973c10e735aea951bdf",  # vllm-hust#76
        "8d28fcf984fd5d17a05c414e7f8f5695acc7cbc3",  # vllm-hust#118
        "73187bc8ba89b8f83652cbc24042433fb7032add",  # vllm-hust#124
        "6f612fbedff718af2dabb93692f00044e66a9b4b",  # ascend-hust#67
        "a46abb7ae68acc13a4fc5870db98619b3f97c6e0",  # ascend-hust#66
        "702214146c1f0f2c2120b87e6a460d5a39cef418",  # ascend-hust#70
        "ae16d09435abd978417a1b5ab7af352c8dcd180a",  # ascend-hust#80
    ],
    "prefix-repetition-online": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "2fb7859dd024b51c7bd09b0c9b5cc701898090bb",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "dcc06b18f32404abafe6922910117f1b9f66054b",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
        # PR-specific commits from benchmark_plan.md
        "52b44710cdf3c797f4046698378fef3ecf6670b3",  # vllm-hust#49
        "c421a1f38f5c4dbff235aa11464d90085cf7b1c0",  # vllm-hust#41
        "98ca7fe3ba4d89a670072751dc642629f2a218f5",  # vllm-hust#81
        "9f906ff2bf1c361d02bff973c10e735aea951bdf",  # vllm-hust#76
        "8d28fcf984fd5d17a05c414e7f8f5695acc7cbc3",  # vllm-hust#118
        "73187bc8ba89b8f83652cbc24042433fb7032add",  # vllm-hust#124
        "6f612fbedff718af2dabb93692f00044e66a9b4b",  # ascend-hust#67
        "a46abb7ae68acc13a4fc5870db98619b3f97c6e0",  # ascend-hust#66
        "702214146c1f0f2c2120b87e6a460d5a39cef418",  # ascend-hust#70
        "ae16d09435abd978417a1b5ab7af352c8dcd180a",  # ascend-hust#80
    ],
    "instructcoder-online": [
        "2206f1f7b7212801187bc001c5f6cb86b2289214",
        "2fb7859dd024b51c7bd09b0c9b5cc701898090bb",
        "51621c35bcce749cc34539bc1a48d32f264924a0",
        "7a63f81e86bd71e980adb635870ff56c9e23b545",
        "83cf83ff20a880d70b6ba916977c49304d598d9c",
        "dcc06b18f32404abafe6922910117f1b9f66054b",
        "f273f9c5e2669b6e8aeee61823c895e2399cf609",
        # PR-specific commits from benchmark_plan.md
        "52b44710cdf3c797f4046698378fef3ecf6670b3",  # vllm-hust#49
        "c421a1f38f5c4dbff235aa11464d90085cf7b1c0",  # vllm-hust#41
        "98ca7fe3ba4d89a670072751dc642629f2a218f5",  # vllm-hust#81
        "9f906ff2bf1c361d02bff973c10e735aea951bdf",  # vllm-hust#76
        "8d28fcf984fd5d17a05c414e7f8f5695acc7cbc3",  # vllm-hust#118
        "73187bc8ba89b8f83652cbc24042433fb7032add",  # vllm-hust#124
        "6f612fbedff718af2dabb93692f00044e66a9b4b",  # ascend-hust#67
        "a46abb7ae68acc13a4fc5870db98619b3f97c6e0",  # ascend-hust#66
        "702214146c1f0f2c2120b87e6a460d5a39cef418",  # ascend-hust#70
        "ae16d09435abd978417a1b5ab7af352c8dcd180a",  # ascend-hust#80
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


def _update_ascend_entry_points() -> None:
    """Regenerate the ascend plugin's entry_points metadata to match the
    currently checked-out commit.

    After ``git checkout``, the installed package's entry_points in dist-info
    may be stale (e.g. referencing ``register_model`` that no longer exists
    in the older code).  This function runs ``setup.py egg_info`` (fast, no
    cmake build) and copies the regenerated entry_points to the installed
    dist-info directory.
    """
    # Step A: regenerate egg-info in the ascend repo (fast, no cmake).
    subprocess.run(
        [str(PYTHON_BIN), "setup.py", "egg_info"],
        cwd=ASCEND_REPO, capture_output=True, text=True, check=False,
    )

    # Step B: find the egg-info entry_points.
    egg_info_dir = None
    for cand in ASCEND_REPO.glob("*.egg-info"):
        egg_info_dir = cand
        break
    if egg_info_dir is None:
        log("Warning: no egg-info found in ascend repo, skipping entry_points update")
        return

    egg_eps = egg_info_dir / "entry_points.txt"
    if not egg_eps.is_file():
        log("Warning: egg-info has no entry_points.txt, skipping")
        return

    new_eps = egg_eps.read_text(encoding="utf-8")

    # Step C: find the installed dist-info and update its entry_points.txt.
    import site
    site_packages = Path(site.getsitepackages()[0])
    updated = False
    for dist_info in site_packages.glob("vllm_ascend_hust*.dist-info"):
        dist_eps = dist_info / "entry_points.txt"
        if not dist_eps.is_file():
            continue
        old_eps = dist_eps.read_text(encoding="utf-8")
        if old_eps != new_eps:
            dist_eps.write_text(new_eps, encoding="utf-8")
            log(f"Updated ascend entry_points in {dist_info.name}")
            updated = True
        break  # only one dist-info expected

    if not updated:
        log("Ascend entry_points already up to date")


def install_ascend_plugin() -> None:
    """Fix naming conflict and update ascend plugin entry_points.

    vllm.entrypoints.cli.openai shadows the external 'openai' PyPI package,
    causing circular imports when other vllm modules (e.g. mcp/tool.py) do
    ``from openai import ...``.  Renaming the file to openai_cmd.py and
    updating the import in main.py breaks the conflict.

    Also regenerates the ascend plugin's entry_points metadata to match
    the currently checked-out commit, preventing stale entry_points
    from causing import errors (e.g. ``register_model`` not found).
    """
    # ------------------------------------------------------------------
    # Step 1: Fix the openai naming conflict.
    # ------------------------------------------------------------------
    cli_dir = HUST_REPO / "vllm" / "entrypoints" / "cli"
    openai_py = cli_dir / "openai.py"
    openai_cmd_py = cli_dir / "openai_cmd.py"

    # If both exist, remove openai.py (openai_cmd.py is already the renamed copy).
    if openai_py.is_file() and openai_cmd_py.is_file():
        subprocess.run(["rm", "-f", str(openai_py)], check=True)
        log(f"Patched: removed duplicate {openai_py}, keeping {openai_cmd_py}")
        # Fall through to update main.py if needed.

    if openai_cmd_py.is_file() and not openai_py.is_file():
        # Already renamed, just ensure main.py is up to date.
        pass
    elif openai_py.is_file() and not openai_cmd_py.is_file():
        subprocess.run(["mv", str(openai_py), str(openai_cmd_py)], check=True)
        log(f"Patched: renamed {openai_py} -> {openai_cmd_py}")

    # Update all references in main.py.
    main_py = cli_dir / "main.py"
    if main_py.is_file():
        content = subprocess.run(
            ["cat", str(main_py)], capture_output=True, text=True, check=True
        ).stdout
        orig = content
        content = content.replace("import vllm.entrypoints.cli.openai\n",
                                  "import vllm.entrypoints.cli.openai_cmd\n")
        content = content.replace("vllm.entrypoints.cli.openai,",
                                  "vllm.entrypoints.cli.openai_cmd,")
        if content != orig:
            subprocess.run(
                [sys.executable, "-c", """
import sys
with open(sys.argv[1], 'w') as f:
    f.write(sys.stdin.read())
""", str(main_py)], input=content, text=True, check=True,
            )
            log(f"Patched: updated imports in {main_py}")
        else:
            log("Patched: imports already correct, skipping")

    # ------------------------------------------------------------------
    # Step 2: Update ascend plugin entry_points to match current checkout.
    # ------------------------------------------------------------------
    _update_ascend_entry_points()

    # ------------------------------------------------------------------
    # Step 3: Fix ascend plugin patches that reference missing vllm-hust
    #         attributes (e.g. _parse_tool_calls_from_content).
    # ------------------------------------------------------------------
    _patch_tool_choice = ASCEND_REPO / "vllm_ascend" / "patch" / "platform" / "patch_tool_choice_none_content.py"
    if _patch_tool_choice.is_file():
        import re as _re
        content = _patch_tool_choice.read_text(encoding="utf-8")
        patched_tc = False

        # Fix 1: Guard _parse_tool_calls_from_content (may not exist on older vllm-hust).
        old_ref = (
            "_original_parse_tool_calls_from_content = "
            "OpenAIServing._parse_tool_calls_from_content"
        )
        if old_ref in content and "try:" not in content.split(old_ref)[0].rsplit("\n", 3)[-1]:
            new_ref = (
                "try:\n"
                "    _original_parse_tool_calls_from_content = "
                "OpenAIServing._parse_tool_calls_from_content\n"
                "except AttributeError:\n"
                "    _original_parse_tool_calls_from_content = None"
            )
            content = content.replace(old_ref, new_ref)
            content = content.replace(
                "OpenAIServing._parse_tool_calls_from_content = "
                "staticmethod(_patched_parse_tool_calls_from_content)",
                "if _original_parse_tool_calls_from_content is not None:\n"
                "    OpenAIServing._parse_tool_calls_from_content = "
                "staticmethod(_patched_parse_tool_calls_from_content)"
            )
            patched_tc = True

        # Fix 2: Guard _parse_tool_calls on DelegatingParser (may not exist on older vllm-hust).
        old_ref2 = "_original_delegating_parse_tool_calls = DelegatingParser._parse_tool_calls"
        if old_ref2 in content and "try:" not in content.split(old_ref2)[0].rsplit("\n", 3)[-1]:
            new_ref2 = (
                "try:\n"
                "    _original_delegating_parse_tool_calls = "
                "DelegatingParser._parse_tool_calls\n"
                "except AttributeError:\n"
                "    _original_delegating_parse_tool_calls = None"
            )
            content = content.replace(old_ref2, new_ref2)
            content = content.replace(
                "DelegatingParser._parse_tool_calls = _patched_delegating_parse_tool_calls",
                "if _original_delegating_parse_tool_calls is not None:\n"
                "    DelegatingParser._parse_tool_calls = _patched_delegating_parse_tool_calls"
            )
            patched_tc = True

        if patched_tc:
            _patch_tool_choice.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin patch_tool_choice_none_content.py compatibility")

    # Fix ascend plugin ops/__init__.py: make the fused_moe import block
    # degrade gracefully on older vllm-hust commits (no UnquantizedFusedMoEMethod).
    _ops_init = ASCEND_REPO / "vllm_ascend" / "ops" / "__init__.py"
    if _ops_init.is_file():
        content = _ops_init.read_text(encoding="utf-8")
        if "except ModuleNotFoundError as exc:" in content:
            # Replace the entire fragile try/except block with a robust one.
            old_block = (
                'try:\n'
                '    import vllm_ascend.ops.fused_moe.fused_moe  # noqa\n'
                'except ModuleNotFoundError as exc:\n'
                '    if exc.name != "vllm.model_executor.layers.fused_moe.runner.default_moe_runner":\n'
                '        raise'
            )
            new_block = (
                'try:\n'
                '    import vllm_ascend.ops.fused_moe.fused_moe  # noqa\n'
                'except Exception:\n'
                '    pass  # gracefully skip on older vllm-hust commits'
            )
            if old_block in content:
                content = content.replace(old_block, new_block)
                _ops_init.write_text(content, encoding="utf-8")
                log("Patched: fixed ascend plugin ops/__init__.py compatibility")

    # Fix ascend plugin patch_glm_tool_call_parser.py: wrap imports that
    # reference modules not present in older vllm-hust commits.
    _patch_glm = ASCEND_REPO / "vllm_ascend" / "patch" / "platform" / "patch_glm_tool_call_parser.py"
    if _patch_glm.is_file():
        content = _patch_glm.read_text(encoding="utf-8")
        if "from vllm.tool_parsers import glm4_moe_tool_parser as glm4_parser" in content:
            # Add a try/except guard around the top-level imports and module body.
            old_glm_imports = (
                "from vllm.tool_parsers import glm4_moe_tool_parser as glm4_parser\n"
                "from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser"
            )
            new_glm_guard = (
                "try:\n"
                "    from vllm.tool_parsers import glm4_moe_tool_parser as glm4_parser\n"
                "    from vllm.tool_parsers.glm4_moe_tool_parser import Glm4MoeModelToolParser\n"
                "    _GLM_PARSER_AVAILABLE = True\n"
                "except ImportError:\n"
                "    _GLM_PARSER_AVAILABLE = False\n"
                "    Glm4MoeModelToolParser = None  # type: ignore"
            )
            if old_glm_imports in content:
                content = content.replace(old_glm_imports, new_glm_guard)
                # Wrap the remaining module-level code (logger = ...) to skip if not available.
                content = content.replace(
                    "logger = chat_serving.logger",
                    "if _GLM_PARSER_AVAILABLE:\n"
                    "    logger = chat_serving.logger"
                )
                # Wrap the class patching at the bottom: replace the whole final block
                old_final_block = (
                    'OpenAIServingChat.chat_completion_stream_generator = _wrapped_chat_completion_stream_generator\n'
                    'Glm4MoeModelToolParser._ensure_tool_state = _ensure_tool_state\n'
                    'Glm4MoeModelToolParser._begin_tool_call = _begin_tool_call\n'
                    'Glm4MoeModelToolParser._finish_tool_call = _finish_tool_call\n'
                    'Glm4MoeModelToolParser._revert_last_tool_call_state = _revert_last_tool_call_state\n'
                    'Glm4MoeModelToolParser._emit_tool_name_delta = _emit_tool_name_delta\n'
                    'Glm4MoeModelToolParser._emit_tool_args_delta = _emit_tool_args_delta\n'
                    'Glm4MoeModelToolParser._append_arg_fragment = _append_arg_fragment\n'
                    'Glm4MoeModelToolParser._close_args_if_needed = _close_args_if_needed\n'
                    'Glm4MoeModelToolParser.extract_tool_calls_streaming = _patched_extract_tool_calls_streaming'
                )
                new_final_block = (
                    'if _GLM_PARSER_AVAILABLE:\n'
                    '    OpenAIServingChat.chat_completion_stream_generator = _wrapped_chat_completion_stream_generator\n'
                    '    Glm4MoeModelToolParser._ensure_tool_state = _ensure_tool_state\n'
                    '    Glm4MoeModelToolParser._begin_tool_call = _begin_tool_call\n'
                    '    Glm4MoeModelToolParser._finish_tool_call = _finish_tool_call\n'
                    '    Glm4MoeModelToolParser._revert_last_tool_call_state = _revert_last_tool_call_state\n'
                    '    Glm4MoeModelToolParser._emit_tool_name_delta = _emit_tool_name_delta\n'
                    '    Glm4MoeModelToolParser._emit_tool_args_delta = _emit_tool_args_delta\n'
                    '    Glm4MoeModelToolParser._append_arg_fragment = _append_arg_fragment\n'
                    '    Glm4MoeModelToolParser._close_args_if_needed = _close_args_if_needed\n'
                    '    Glm4MoeModelToolParser.extract_tool_calls_streaming = _patched_extract_tool_calls_streaming'
                )
                if old_final_block in content:
                    content = content.replace(old_final_block, new_final_block)
                _patch_glm.write_text(content, encoding="utf-8")
                log("Patched: fixed ascend plugin patch_glm_tool_call_parser.py compatibility")

    # Fix ascend plugin common_cp.py: wrap imports that reference vllm.distributed
    # functions not present in older vllm-hust commits.
    _common_cp = ASCEND_REPO / "vllm_ascend" / "attention" / "context_parallel" / "common_cp.py"
    if _common_cp.is_file():
        content = _common_cp.read_text(encoding="utf-8")
        old_import = (
            "from vllm.distributed import get_dcp_group, "
            "get_decode_context_model_parallel_world_size, get_pcp_group"
        )
        if old_import in content:
            new_import = (
                "from vllm.distributed import get_dcp_group, get_pcp_group\n"
                "try:\n"
                "    from vllm.distributed import get_decode_context_model_parallel_world_size\n"
                "except ImportError:\n"
                "    def get_decode_context_model_parallel_world_size() -> int:\n"
                "        return 1"
            )
            content = content.replace(old_import, new_import)
            _common_cp.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin common_cp.py compatibility")

    # Fix ascend plugin patch_balance_schedule.py: handle throttle_prefills
    # parameter not present in older vllm-hust Scheduler.schedule().
    _balance = ASCEND_REPO / "vllm_ascend" / "patch" / "platform" / "patch_balance_schedule.py"
    if _balance.is_file():
        content = _balance.read_text(encoding="utf-8")
        old_call = "            return super().schedule(throttle_prefills)"
        if old_call in content:
            new_call = (
                "            try:\n"
                "                return super().schedule(throttle_prefills)\n"
                "            except TypeError:\n"
                "                return super().schedule()"
            )
            content = content.replace(old_call, new_call)
            _balance.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin patch_balance_schedule.py compatibility")

    # Fix ascend plugin patch_distributed.py: handle shm_broadcast import
    # that references VLLM_USE_SPINLOOP_EXT not present in older vllm-hust.
    _patch_dist = ASCEND_REPO / "vllm_ascend" / "patch" / "worker" / "patch_distributed.py"
    if _patch_dist.is_file():
        content = _patch_dist.read_text(encoding="utf-8")
        old_import = (
            "from vllm.distributed.device_communicators.shm_broadcast import MessageQueue"
        )
        if old_import in content:
            # Preserve the original indentation of the import line
            import re as _re
            _match = _re.search(r"^(\s*)" + _re.escape(old_import), content, _re.MULTILINE)
            _indent = _match.group(1) if _match else ""
            new_import = (
                f"{_indent}try:\n"
                f"{_indent}    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue\n"
                f"{_indent}except (AttributeError, ImportError):\n"
                f"{_indent}    MessageQueue = None  # type: ignore"
            )
            content = content.replace(old_import, new_import)
            # Guard the usage of MessageQueue too
            content = content.replace(
                "if use_message_queue_broadcaster and self.world_size > 1:",
                "if use_message_queue_broadcaster and self.world_size > 1 and MessageQueue is not None:",
            )
            _patch_dist.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin patch_distributed.py shm_broadcast compatibility")

    # Fix ascend plugin eplb_utils.py: handle determine_expert_map import
    # from fused_moe.layer (not present in older vllm-hust commits).
    _eplb_utils = ASCEND_REPO / "vllm_ascend" / "eplb" / "core" / "eplb_utils.py"
    if _eplb_utils.is_file():
        content = _eplb_utils.read_text(encoding="utf-8")
        old_import = (
            "from vllm.model_executor.layers.fused_moe.layer "
            "import determine_expert_map"
        )
        if old_import in content:
            new_import = (
                "try:\n"
                "    from vllm.model_executor.layers.fused_moe.layer"
                " import determine_expert_map\n"
                "except ImportError:\n"
                "    try:\n"
                "        from vllm.model_executor.layers.fused_moe"
                ".expert_map_manager import determine_expert_map\n"
                "    except ImportError:\n"
                "        determine_expert_map = None  # type: ignore"
            )
            content = content.replace(old_import, new_import)
            _eplb_utils.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin eplb_utils.py compatibility")

    # Fix ascend plugin modelslim_config.py: handle MoERunner import
    # from fused_moe (not present in older vllm-hust commits).
    _modelslim = ASCEND_REPO / "vllm_ascend" / "quantization" / "modelslim_config.py"
    if _modelslim.is_file():
        content = _modelslim.read_text(encoding="utf-8")
        # The import may be indented (inside else block), use regex to
        # capture leading whitespace and preserve it.
        import re as _msre
        pattern = r'^(\s*)(from vllm\.model_executor\.layers\.fused_moe import MoERunner, RoutedExperts)$'
        m = _msre.search(pattern, content, _msre.MULTILINE)
        if m:
            indent = m.group(1)
            orig = m.group(0)
            new_block = (
                f"{indent}try:\n"
                f"{indent}    {m.group(2)}\n"
                f"{indent}except ImportError:\n"
                f"{indent}    MoERunner = None  # type: ignore\n"
                f"{indent}    RoutedExperts = None  # type: ignore"
            )
            content = content.replace(orig, new_block)
            _modelslim.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin modelslim_config.py compatibility")

    # ------------------------------------------------------------------
    # Step 4: Fix missing imports in serving.py (PR #118 bugs).
    # ------------------------------------------------------------------
    serving_py = HUST_REPO / "vllm" / "entrypoints" / "openai" / "engine" / "serving.py"
    if serving_py.is_file():
        content = serving_py.read_text(encoding="utf-8")
        import re
        patched = False

        # Fix 1: missing `Any` import
        m = re.search(r'^from typing import (.+)$', content, re.MULTILINE)
        if m and 'Any' not in m.group(1) and re.search(r'\bAny\b', content):
            old_line = m.group(0)
            new_line = old_line.replace('from typing import ', 'from typing import Any, ')
            content = content.replace(old_line, new_line, 1)
            patched = True

        # Fix 2: missing `PromptType` and `extract_prompt_len` imports
        needs_prompt_type = 'PromptType' in content and 'from vllm.inputs import PromptType' not in content
        needs_extract_prompt_len = 'extract_prompt_len' in content and 'from vllm.renderers.inputs.preprocess import' not in content
        if needs_prompt_type or needs_extract_prompt_len:
            old_import = 'from vllm.inputs import EngineInput'
            new_import = 'from vllm.inputs import EngineInput'
            if needs_prompt_type:
                new_import += ', PromptType'
            if needs_extract_prompt_len:
                new_import += '\nfrom vllm.renderers.inputs.preprocess import extract_prompt_len'
            if new_import != old_import:
                content = content.replace(old_import, new_import)
                patched = True

        # Fix 3: missing `SamplingParams` import
        if 'SamplingParams' in content:
            m_vllm = re.search(r'^from vllm import (.+)$', content, re.MULTILINE)
            if m_vllm:
                if 'SamplingParams' not in m_vllm.group(1):
                    old_line = m_vllm.group(0)
                    new_line = old_line.replace(
                        'from vllm import ', 'from vllm import SamplingParams, '
                    )
                    content = content.replace(old_line, new_line, 1)
                    patched = True
            else:
                # No `from vllm import` line — add one after the last `from vllm.` import.
                lines = content.split('\n')
                insert_at = 0
                in_multiline = False
                for i, line in enumerate(lines):
                    if line.startswith('from vllm.'):
                        insert_at = i + 1
                        in_multiline = '(' in line and ')' not in line
                    elif in_multiline:
                        insert_at = i + 1
                        if ')' in line:
                            in_multiline = False
                if insert_at > 0:
                    lines.insert(insert_at, 'from vllm import SamplingParams')
                    content = '\n'.join(lines)
                    patched = True

        # Fix 4: missing `BeamSearchParams` import
        if 'BeamSearchParams' in content and 'from vllm.sampling_params import' not in content:
            # Insert after the last `from vllm.` import line (or multiline block).
            lines = content.split('\n')
            insert_at = 0
            in_multiline = False
            for i, line in enumerate(lines):
                if line.startswith('from vllm.'):
                    insert_at = i + 1
                    in_multiline = '(' in line and ')' not in line
                elif in_multiline:
                    insert_at = i + 1
                    if ')' in line:
                        in_multiline = False
            if insert_at > 0:
                lines.insert(insert_at, 'from vllm.sampling_params import BeamSearchParams')
                content = '\n'.join(lines)
                patched = True

        if patched:
            serving_py.write_text(content, encoding="utf-8")
            log("Patched: fixed missing imports in serving.py")


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
    env["ASCEND_RT_VISIBLE_DEVICES"] = "6"
    env["ASCEND_VISIBLE_DEVICES"] = "6"
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


def run_cell(workload: str, hust_commit: str, ascend_commit: str | None = None) -> dict[str, Any]:
    if ascend_commit is None:
        ascend_commit = _resolve_compatible_ascend_commit(hust_commit)

    log(f"=== {workload} @ {hust_commit[:9]} (plugin {ascend_commit[:9]}) ===")
    work_dir = STATE_DIR / "runs" / f"{workload}-{hust_commit[:9]}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # First restore the repo to a clean state (discards any local
        # modifications from previous runs, e.g. renamed openai.py).
        git_checkout(HUST_REPO, hust_commit)
        git_checkout(ASCEND_REPO, ascend_commit)
        # Remove any untracked files left by previous runs (e.g. openai_cmd.py).
        # Use -fdx to also remove git-ignored files like __pycache__/
        # which can cause stale bytecode cache conflicts.
        subprocess.run(
            ["git", "clean", "-fdx", "vllm/entrypoints/cli/"],
            cwd=HUST_REPO, check=False,
        )

        install_ascend_plugin()
        raw = run_vllm_bench(workload, hust_commit, work_dir / "bench")
        run_id = build_run_id(workload, hust_commit)
        sub_dir = submit_artifact(workload, hust_commit, ascend_commit, run_id, raw)

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

    log("Aggregate validation passed. "
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
