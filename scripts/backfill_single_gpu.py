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

Python interpreter discovery
----------------------------
If the script is invoked with a bare ``python3`` that does not have the
``vllm_hust_benchmark`` package available, it will re-execute itself
using the interpreter discovered below (``BACKFILL_PYTHON`` env var,
``~/miniconda3/envs/vllm-hust-dev/bin/python``, or ``sys.executable``).

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

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-re-execute with the correct Python interpreter.
# User runs e.g. ``python3 scripts/backfill_single_gpu.py plan``, but the
# ``vllm_hust_benchmark`` package lives in the ``vllm-hust-dev`` conda env.
# If the current interpreter cannot import the package, re-exec with the
# ``BACKFILL_PYTHON`` (or discovered) interpreter.
# ---------------------------------------------------------------------------
_EXPECTED_PYTHON = Path(
    os.environ.get("BACKFILL_PYTHON")
    or Path.home() / "miniconda3/envs/vllm-hust-dev/bin/python"
)
if not _EXPECTED_PYTHON.is_file():
    _EXPECTED_PYTHON = Path(sys.executable)

try:
    # Quick check: can we import the package?
    import vllm_hust_benchmark  # noqa: F401
except ImportError:
    if sys.executable != str(_EXPECTED_PYTHON) and _EXPECTED_PYTHON.is_file():
        # Re-execute this script with the correct Python interpreter.
        os.execv(str(_EXPECTED_PYTHON), [str(_EXPECTED_PYTHON)] + sys.argv)
    # No fallback available — let the normal import error surface below.

import argparse
import hashlib
import json
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

from vllm_hust_benchmark.leaderboard_exclusions import (
    load_leaderboard_exclusions,
    match_leaderboard_exclusion,
)
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


def _resolve_compatible_ascend_commit(hust_commit: str) -> str:
    """Resolve the latest vllm-ascend-hust origin/main commit.

    Previously this was hardcoded to ``bf2984e...`` which was excluded from
    leaderboard aggregation (see docs/leaderboard-exclusions.json).  By
    using the latest ``origin/main`` instead, backfill submissions are no
    longer rejected by the exclusion check.

    If ``origin/main`` is not reachable (e.g. offline), falls back to the
    current HEAD of the local ascend repo.

    The ``install_ascend_plugin()`` function applies comprehensive
    compatibility patches (try/except ImportError guards) so that the
    latest ascend code works with older vllm-hust commits passed to
    ``run_cell``.

    Use the ``--ascend-commit`` CLI flag to override this auto-detection
    with a specific commit.
    """
    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "origin/main"],
        cwd=ASCEND_REPO, capture_output=True, text=True, check=False,
    )
    sha = out.stdout.strip()
    if sha and len(sha) == 40:
        return sha
    # Fallback: use current HEAD if origin/main is not accessible.
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ASCEND_REPO, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def _derive_engine_version(hust_repo: Path, hust_commit: str) -> str:
    """Derive the vllm-hust engine version from the given commit.

    Uses ``git describe --tags --always --long --abbrev=9 <commit>`` to
    produce a consistent version string like::

        0.17.2.post1-1357-g83cf83ff2

    The output is normalised:

    * ``--long`` guarantees the commit count is always present, so every
      version carries the same granularity.
    * ``.dirty`` / ``-dirty`` suffixes are stripped because the benchmark
      results were produced by a *clean* checkout of that commit, and the
      dirty marker only reflects the state of the repo at backfill time,
      which is irrelevant to the result.

    Falls back to ``0.0.0.dev0+g{short_commit}`` (PEP 440 compatible) if
    ``git describe`` returns an empty string.
    """
    try:
        out = subprocess.run(
            ["git", "describe", "--tags", "--always", "--long", "--abbrev=9", hust_commit],
            cwd=hust_repo, capture_output=True, text=True, check=False,
        )
        version = out.stdout.strip()
        if version:
            # Strip .dirty / -dirty suffix — the result was produced by a
            # clean checkout of this commit, and the dirty flag only
            # reflects the state of the backfill repo, not the result.
            version = re.sub(r'[.-]dirty$', '', version, flags=re.IGNORECASE)
            return version
    except OSError:
        pass
    return f"0.0.0.dev0+g{hust_commit[:9]}"


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


def commit_exists(repo: Path, commit: str) -> bool:
    """Check whether a commit (full or short SHA) exists in the repo."""
    r = subprocess.run(
        ["git", "cat-file", "-t", commit],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    return r.returncode == 0


def commit_on_main_branch(repo: Path, commit: str) -> bool:
    """Check whether a commit is an ancestor of origin/main."""
    if not commit_exists(repo, commit):
        return False
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=repo, capture_output=True, text=True, check=False,
    )
    return r.returncode == 0


def select_idle_npu() -> int | None:
    """Query ``npu-smi info`` and return the first idle NPU device index.

    An NPU is considered idle when its HBM usage is below a threshold
    (default 1000 MB).  Returns ``None`` if no idle NPU is found or if
    ``npu-smi`` is not available.
    """
    try:
        output = subprocess.check_output(
            ["npu-smi", "info"], text=True, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        log("Warning: npu-smi not available, cannot detect idle NPU")
        return None

    idle_threshold_mb = 5000
    lines = output.splitlines()
    # Format: each NPU has 2 data lines:
    #   | N  Name     | Health | Power  Temp  Hugepages |
    #   | N  Bus-Id   | AICore | Mem-Usage  HBM-Usage  |
    # HBM-Usage is on the chip-info line (2nd line) as "used/total"
    for i, line in enumerate(lines):
        m = re.match(r"^\|\s*(\d+)\s+\S+\s+\|", line)
        if not m:
            continue
        # This is an NPU header line; next line has HBM-Usage
        if i + 1 < len(lines):
            chip_line = lines[i + 1]
            hbm_m = re.search(r"(\d+)\s*/\s*(\d+)\s*\|\s*$", chip_line)
            if hbm_m:
                used = int(hbm_m.group(1))
                if used < idle_threshold_mb:
                    return int(m.group(1))
    return None


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
    # Step 2.5: Regenerate _build_info.py in the ascend plugin.
    #
    # _build_info.py is gitignored in vllm-ascend-hust (generated by
    # setup.py's gen_build_info()).  After git checkout, the file is
    # missing, which causes ``ImportError: cannot import name '_build_info'
    # from 'vllm_ascend'`` at runtime.  Running ``setup.py build_py``
    # regenerates it without triggering cmake compilation.
    # ------------------------------------------------------------------
    _build_info_py = ASCEND_REPO / "vllm_ascend" / "_build_info.py"
    if not _build_info_py.is_file():
        _bi = subprocess.run(
            [str(PYTHON_BIN), "setup.py", "build_py"],
            cwd=ASCEND_REPO, capture_output=True, text=True, check=False,
        )
        if _build_info_py.is_file():
            log(f"Regenerated {_build_info_py}")
        else:
            log(f"Warning: failed to regenerate _build_info.py: {_bi.stderr.strip()}")
    else:
        log("_build_info.py already exists, skipping")

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

    # Fix 4: Guard ops/__init__.py imports that reference missing vllm modules.
    _ops_init = ASCEND_REPO / "vllm_ascend" / "ops" / "__init__.py"
    if _ops_init.is_file():
        content = _ops_init.read_text(encoding="utf-8")
        patched_ops = False
        # Guard import of expert_map_manager (not present in older vllm-hust).
        old_ref = "from vllm.distributed.utils import expert_map_manager"
        if old_ref in content:
            content = content.replace(
                old_ref,
                "try:\n    " + old_ref + "\nexcept ImportError:\n    expert_map_manager = None",
            )
            patched_ops = True
        if patched_ops:
            _ops_init.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin ops/__init__.py compatibility")

    # Fix 5: Guard patch_glm_tool_call_parser.py (may reference missing imports).
    _glm_parser = ASCEND_REPO / "vllm_ascend" / "patch" / "platform" / "patch_glm_tool_call_parser.py"
    if _glm_parser.is_file():
        content = _glm_parser.read_text(encoding="utf-8")
        patched_glm = False
        # Guard import of _parse_tool_calls_from_content
        old_ref = "from vllm.entrypoints.openai.serving import OpenAIServing"
        if old_ref in content:
            new_ref = (
                "try:\n"
                "    from vllm.entrypoints.openai.serving import OpenAIServing\n"
                "except ImportError:\n"
                "    OpenAIServing = None"
            )
            content = content.replace(old_ref, new_ref)
            # Also guard the usage
            content = content.replace(
                "_original_parse_tool_calls_from_content = OpenAIServing._parse_tool_calls_from_content",
                "if OpenAIServing is not None:\n"
                "    _original_parse_tool_calls_from_content = OpenAIServing._parse_tool_calls_from_content\n"
                "else:\n"
                "    _original_parse_tool_calls_from_content = None",
            )
            patched_glm = True
        if patched_glm:
            _glm_parser.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin patch_glm_tool_call_parser.py compatibility")

    # Fix 6: Guard common_cp.py imports.
    _common_cp = ASCEND_REPO / "vllm_ascend" / "patch" / "platform" / "common_cp.py"
    if _common_cp.is_file():
        content = _common_cp.read_text(encoding="utf-8")
        patched_cp = False
        old_ref = "from vllm.distributed.utils import expert_map_manager"
        if old_ref in content:
            content = content.replace(
                old_ref,
                "try:\n    " + old_ref + "\nexcept ImportError:\n    expert_map_manager = None",
            )
            patched_cp = True
        if patched_cp:
            _common_cp.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin common_cp.py compatibility")

    # Fix 7: Guard patch_distributed.py shm_broadcast import.
    _patch_dist = ASCEND_REPO / "vllm_ascend" / "patch" / "platform" / "patch_distributed.py"
    if _patch_dist.is_file():
        content = _patch_dist.read_text(encoding="utf-8")
        patched_dist = False

        # Guard shm_broadcast import.
        old_ref = "from vllm.distributed.device_communicators.shm_broadcast import MessageQueue"
        if old_ref in content:
            new_ref = (
                "try:\n"
                "    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue\n"
                "except (ImportError, AttributeError):\n"
                "    MessageQueue = None"
            )
            # Find the indentation of the original line.
            lines = content.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped == old_ref:
                    indent = line[: len(line) - len(stripped)]
                    indented_new = "\n".join(
                        indent + l if l.strip() else l
                        for l in new_ref.split("\n")
                    )
                    lines[i] = indented_new
                    content = "\n".join(lines)
                    patched_dist = True
                    break

        if patched_dist:
            _patch_dist.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin patch_distributed.py shm_broadcast compatibility")

    # Fix 8: Guard eplb_utils.py imports.
    _eplb_utils = ASCEND_REPO / "vllm_ascend" / "eplb" / "eplb_utils.py"
    if _eplb_utils.is_file():
        content = _eplb_utils.read_text(encoding="utf-8")
        patched_eplb = False
        old_ref = "from vllm.distributed.utils import expert_map_manager"
        if old_ref in content:
            content = content.replace(
                old_ref,
                "try:\n    " + old_ref + "\nexcept ImportError:\n    expert_map_manager = None",
            )
            patched_eplb = True
        if patched_eplb:
            _eplb_utils.write_text(content, encoding="utf-8")
            log("Patched: fixed ascend plugin eplb_utils.py compatibility")

    # Fix 9: Add missing imports in serving.py (may be needed by older vllm-hust).
    _serving = ASCEND_REPO / "vllm_ascend" / "patch" / "platform" / "serving.py"
    if _serving.is_file():
        content = _serving.read_text(encoding="utf-8")
        patched_serving = False
        missing_imports = (
            "from vllm.entrypoints.openai.serving import OpenAIServing\n"
        )
        if missing_imports.strip() not in content:
            content = missing_imports + content
            patched_serving = True
        if patched_serving:
            _serving.write_text(content, encoding="utf-8")
            log("Patched: fixed missing imports in serving.py")

    # Fix 10: Add missing ``Any`` import in vllm-hust's serving.py (commit 8d28fcf98).
    _hust_serving = HUST_REPO / "vllm" / "entrypoints" / "openai" / "engine" / "serving.py"
    if _hust_serving.is_file():
        content = _hust_serving.read_text(encoding="utf-8")
        if "from typing import Any" not in content and "Any" in content:
            # Find the last typing import line and add Any.
            lines = content.split("\n")
            last_typing_import = -1
            for i, line in enumerate(lines):
                if line.strip().startswith("from typing import") or line.strip().startswith("import typing"):
                    last_typing_import = i
            if last_typing_import >= 0:
                ti = lines[last_typing_import]
                if "Any" not in ti:
                    lines[last_typing_import] = ti.rstrip() + ", Any"
                    _hust_serving.write_text("\n".join(lines), encoding="utf-8")
                    log("Patched: added missing Any import in vllm-hust serving.py")
                else:
                    log("Patched: Any import already present in vllm-hust serving.py")
            else:
                # No typing import exists at all; add one.
                lines.insert(0, "from typing import Any")
                _hust_serving.write_text("\n".join(lines), encoding="utf-8")
                log("Patched: added from typing import Any to vllm-hust serving.py")

    # Fix 11: Guard glm4_moe_tool_parser import in patch_glm_tool_call_parser.py.
    _glm_parser = ASCEND_REPO / "vllm_ascend" / "patch" / "platform" / "patch_glm_tool_call_parser.py"
    if _glm_parser.is_file():
        content = _glm_parser.read_text(encoding="utf-8")
        patched_glm2 = False
        old_ref = "from vllm.tool_parsers import glm4_moe_tool_parser as glm4_parser"
        if old_ref in content:
            new_ref = (
                "try:\n"
                "    from vllm.tool_parsers import glm4_moe_tool_parser as glm4_parser\n"
                "except ImportError:\n"
                "    glm4_parser = None"
            )
            # Find the indentation of the original line.
            lines = content.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped == old_ref:
                    indent = line[:len(line) - len(stripped)]
                    indented_new = "\n".join(
                        indent + l if l.strip() else l
                        for l in new_ref.split("\n")
                    )
                    lines[i] = indented_new
                    content = "\n".join(lines)
                    patched_glm2 = True
                    break
        if patched_glm2:
            _glm_parser.write_text(content, encoding="utf-8")
            log("Patched: guarded glm4_moe_tool_parser import in patch_glm_tool_call_parser.py")

    # Fix 12: Guard moe_runtime_args.py circular import of vllm_ascend.ops.
    _moe_runtime = ASCEND_REPO / "vllm_ascend" / "ops" / "fused_moe" / "moe_runtime_args.py"
    if _moe_runtime.is_file():
        content = _moe_runtime.read_text(encoding="utf-8")
        patched_moe = False
        old_ref = "import vllm_ascend.ops.fused_moe.moe_stage_params as _stage_params"
        if old_ref in content:
            new_ref = (
                "try:\n"
                "    import vllm_ascend.ops.fused_moe.moe_stage_params as _stage_params\n"
                "except ImportError:\n"
                "    _stage_params = None"
            )
            lines = content.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped == old_ref:
                    indent = line[:len(line) - len(stripped)]
                    indented_new = "\n".join(
                        indent + l if l.strip() else l
                        for l in new_ref.split("\n")
                    )
                    lines[i] = indented_new
                    content = "\n".join(lines)
                    patched_moe = True
                    break
        if patched_moe:
            _moe_runtime.write_text(content, encoding="utf-8")
            log("Patched: guarded moe_runtime_args.py import of moe_stage_params")

    # Fix 13: Add missing imports in vllm-hust's serving.py for old commits.
    # Old commits (e.g. 8d28fcf98, 73187bc8b) are missing several imports:
    #
    #   - ``PromptType`` from ``vllm.inputs`` (used in ``_extract_prompt_components``)
    #   - ``SamplingParams``, ``BeamSearchParams`` from ``vllm.sampling_params``
    #     (used in ``_extract_prompt_components``)
    #
    # The old ``main.py`` eagerly imports all CLI modules (including ``bench``
    # subcommands), so these missing imports block ALL subcommands.
    _hust_serving = HUST_REPO / "vllm" / "entrypoints" / "openai" / "engine" / "serving.py"
    if _hust_serving.is_file():
        content = _hust_serving.read_text(encoding="utf-8")
        patched_serving = False
        lines = content.split("\n")

        # 1) Fix existing import lines that are missing names.
        #    e.g. ``from vllm.inputs import EngineInput`` -> add ``PromptType``.
        import_map = {
            "from vllm.inputs import": {"PromptType"},
        }
        for i, line in enumerate(lines):
            stripped = line.strip()
            for prefix, needed_names in import_map.items():
                if stripped.startswith(prefix):
                    missing = needed_names - set(stripped.split())
                    if missing:
                        lines[i] = line.rstrip() + ", " + ", ".join(sorted(missing))
                        patched_serving = True
                    break

        # 2) Add entirely new import lines that are missing.
        #    The old commit doesn't have ``from vllm.sampling_params import ...``.
        #    Only consider root-level imports (no indentation) to avoid inserting
        #    inside a method body.  Also handle multi-line imports correctly.
        new_imports = [
            "from vllm.sampling_params import BeamSearchParams, SamplingParams",
        ]
        for new_import in new_imports:
            prefix = new_import.split(" import ")[0] + " import "
            if any(prefix in line for line in lines):
                continue
            # Find the last line of the root-level import block by scanning for
            # root-level ``from vllm.`` imports and all their continuation lines
            # (including the closing ``)`` of multi-line imports, even when the
            # ``)`` is not indented).
            #
            # The import block ends when we hit a non-``from``, non-``)``,
            # non-indented line that is not a blank line.
            insert_after = -1
            seen_root_import = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("from vllm.") and not line.startswith((" ", "\t")):
                    seen_root_import = True
                    insert_after = i
                elif seen_root_import and (line.startswith((" ", "\t")) or stripped == ")"):
                    # Continuation of multi-line import (indented items or closing paren).
                    insert_after = i
                elif seen_root_import:
                    # Non-import, non-continuation line → import block ended.
                    break
            lines.insert(insert_after + 1, new_import)
            patched_serving = True

        if patched_serving:
            content = "\n".join(lines)
            _hust_serving.write_text(content, encoding="utf-8")
            log("Patched: added missing imports in vllm-hust serving.py")

    # Fix 14: Create missing ``vllm/beam_search/`` package.
    #
    # ``vllm/entrypoints/llm.py`` imports from ``vllm.beam_search``:
    #
    #   from vllm.beam_search import (
    #       BeamSearchInstance,
    #       BeamSearchOutput,
    #       BeamSearchSequence,
    #       create_sort_beams_key_function,
    #   )
    #
    # But the ``vllm/beam_search/`` directory doesn't exist in the old
    # commit.  The actual beam_search code is at
    # ``vllm/entrypoints/generate/beam_search/``.  We create the missing
    # package that re-exports the needed symbols.
    _beam_search_dir = HUST_REPO / "vllm" / "beam_search"
    if not _beam_search_dir.is_dir():
        _beam_search_dir.mkdir(parents=True, exist_ok=True)
        _init_py = _beam_search_dir / "__init__.py"
        _init_py.write_text(
            "# Auto-generated by backfill_single_gpu.py for compatibility\n"
            "# with older vllm-hust commits that reference vllm.beam_search\n"
            "# but do not have the package.\n"
            "from vllm.entrypoints.generate.beam_search.utils import (\n"
            "    BeamSearchInstance,\n"
            "    BeamSearchOutput,\n"
            "    BeamSearchSequence,\n"
            "    create_sort_beams_key_function,\n"
            ")\n"
            "\n"
            "__all__ = [\n"
            '    "BeamSearchInstance",\n'
            '    "BeamSearchOutput",\n'
            '    "BeamSearchSequence",\n'
            '    "create_sort_beams_key_function",\n'
            "]\n",
            encoding="utf-8",
        )
        log("Patched: created missing vllm/beam_search/ package for old commit compatibility")

    # Fix 15: Guard ``OnlineQuantizationConfigArgs`` import in llm.py.
    #
    # The old commit (8d28fcf98) has ``vllm/entrypoints/llm.py`` importing
    # ``OnlineQuantizationConfigArgs`` from ``vllm.config.quantization``,
    # but the class doesn't exist in the old commit's quantization.py
    # (only ``QuantizationConfigArgs`` and ``QuantSpec`` exist).
    #
    # This causes:
    #   ImportError: cannot import name 'OnlineQuantizationConfigArgs'
    #   from 'vllm.config.quantization'
    #
    # We guard the import with try/except and replace the type annotation
    # with ``Any`` when the class is missing.
    _llm_py = HUST_REPO / "vllm" / "entrypoints" / "llm.py"
    if _llm_py.is_file():
        content = _llm_py.read_text(encoding="utf-8")
        patched_llm = False

        # Guard the import.
        old_import = "from vllm.config.quantization import (\n    OnlineQuantizationConfigArgs,\n)"
        new_import = (
            "try:\n"
            "    from vllm.config.quantization import (\n"
            "        OnlineQuantizationConfigArgs,\n"
            "    )\n"
            "except ImportError:\n"
            "    OnlineQuantizationConfigArgs = None  # type: ignore[assignment]"
        )
        if old_import in content:
            content = content.replace(old_import, new_import)
            patched_llm = True

        # Replace the type annotation ``| OnlineQuantizationConfigArgs | None``
        # with ``| Any | None`` when the class may be None.
        # The annotation is on the ``quantization_config`` parameter.
        old_usage = "        | OnlineQuantizationConfigArgs\n        | None = None,"
        new_usage = "        | Any\n        | None = None,"
        if old_usage in content:
            content = content.replace(old_usage, new_usage)
            patched_llm = True

        if patched_llm:
            _llm_py.write_text(content, encoding="utf-8")
            log("Patched: guarded OnlineQuantizationConfigArgs import in llm.py")

    # Fix 16: Create missing ``vllm/entrypoints/utils.py`` module.
    #
    # ``vllm/entrypoints/llm.py`` imports from ``vllm.entrypoints.utils``:
    #
    #   from vllm.entrypoints.utils import log_non_default_args
    #
    # But the ``vllm/entrypoints/utils.py`` file doesn't exist in the old
    # commit.  The actual function is in
    # ``vllm/entrypoints/serve/utils/api_utils.py``.  We create the missing
    # module that re-exports the needed function.
    _entrypoints_utils = HUST_REPO / "vllm" / "entrypoints" / "utils.py"
    if not _entrypoints_utils.is_file():
        _entrypoints_utils.write_text(
            "# Auto-generated by backfill_single_gpu.py for compatibility\n"
            "# with older vllm-hust commits that reference this module.\n"
            "from vllm.entrypoints.serve.utils.api_utils import log_non_default_args\n"
            "\n"
            "__all__ = [\"log_non_default_args\"]\n",
            encoding="utf-8",
        )
        log("Patched: created missing vllm/entrypoints/utils.py module")


# ---------------------------------------------------------------------------
# Existing-cell discovery
# ---------------------------------------------------------------------------

def load_leaderboard() -> list[dict[str, Any]]:
    snapshot = REPO_ROOT / "leaderboard-data" / "snapshots" / "leaderboard_single.json"
    if not snapshot.is_file():
        return []
    return json.loads(snapshot.read_text(encoding="utf-8"))


def _data_file_commits() -> set[str]:
    """Return the set of unique full 40-char git_commit SHAs from the
    leaderboard_single.json data file.  Used by ``plan`` to determine
    the authoritative list of commits to check for missing cells."""
    leaderboard = load_leaderboard()
    commits: set[str] = set()
    for entry in leaderboard:
        gc = entry.get("metadata", {}).get("git_commit", "") or ""
        if len(gc) == 40:
            commits.add(gc)
    return commits


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
        if arg == "--max-model-len":
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


def _build_env(npu_id: int = 0) -> dict[str, str]:
    """Build the environment for running vllm commands.

    *npu_id* – the NPU device index to use (default 0).
    """
    env = os.environ.copy()
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("VLLM_USE_V1", "1")
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(npu_id)
    env["ASCEND_VISIBLE_DEVICES"] = str(npu_id)
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

    # Set VLLM_VERSION so the ascend plugin can parse the vllm version.
    # Old commits (pre-v0.23) may have __version__ == "dev" when _version.py
    # is missing after git checkout, causing get_vllm_upstream_version() to
    # raise ValueError("Invalid vllm version dev").
    env.setdefault("VLLM_VERSION", "0.17.0")
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
        "--max-model-len", "30720",
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
    workload: str, hust_commit: str, output_dir: Path, npu_id: int = 0,
) -> Path:
    """Run the right vllm bench subcommand and return the raw result JSON path."""
    params = SCENARIO_PARAMS[workload]
    benchmark_type = params["benchmark_type"]
    output_dir.mkdir(parents=True, exist_ok=True)

    bench_log = output_dir / "bench.log"
    env = _build_env(npu_id=npu_id)

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
            "--max-model-len", "30720",
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
            "--max-model-len", "30720",
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
        # Check the SAME_SPEC_GPU_MEMORY_UTILIZATION env var first, then
        # fall back to the hardcoded 0.6 used by the v0.18.0 baseline.
        gpu_mem_override = os.environ.get("SAME_SPEC_GPU_MEMORY_UTILIZATION", "").strip()
        if gpu_mem_override:
            try:
                server_params["gpu_memory_utilization"] = float(gpu_mem_override)
            except ValueError:
                server_params["gpu_memory_utilization"] = 0.6
        else:
            server_params["gpu_memory_utilization"] = 0.6
    if "max_model_len" not in server_params:
        # Check the SAME_SPEC_MAX_MODEL_LEN env var, which the official
        # build_same_spec_payload reads via _maybe_apply_max_model_len_override.
        # Always include max_model_len so that the hash is consistent — the
        # benchmark run commands also pass --max-model-len explicitly.
        max_model_len_override = os.environ.get("SAME_SPEC_MAX_MODEL_LEN", "").strip()
        if max_model_len_override:
            try:
                server_params["max_model_len"] = int(max_model_len_override)
            except ValueError:
                server_params["max_model_len"] = 30720
        else:
            server_params["max_model_len"] = 30720

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
            "--max-model-len", "30720",
            "--output-json", str(raw_path),
        ]
    elif benchmark_type == "throughput":
        parts = [
            str(PYTHON_BIN), "-m", "vllm.entrypoints.cli.main", "bench", "throughput",
            "--model", MODEL_NAME,
            "--dataset-name", params["dataset_name"],
            "--num-prompts", str(params["num_prompts"]),
            "--gpu-memory-utilization", "0.6",
            "--max-model-len", "30720",
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
    workload: str, hust_commit: str, ascend_commit: str, run_id: str, raw: Path,
    *,
    engine_version: str | None = None,
) -> Path:
    if engine_version is None:
        engine_version = _derive_engine_version(HUST_REPO, hust_commit)

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
        "--engine-version", engine_version,
        "--core-version", engine_version,
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


def run_cell(workload: str, hust_commit: str, ascend_commit: str | None = None,
             npu_id: int = 0) -> dict[str, Any]:
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
        # Use -fdx on HUST_REPO to also remove git-ignored files like __pycache__/
        # which can cause stale bytecode cache conflicts.
        subprocess.run(
            ["git", "clean", "-fdx", "vllm/entrypoints/cli/"],
            cwd=HUST_REPO, check=False,
        )

        # Apply compatibility patches for older vllm-hust commits.
        install_ascend_plugin()

        raw = run_vllm_bench(workload, hust_commit, work_dir / "bench", npu_id=npu_id)
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
    commits = sorted(_data_file_commits())
    if not commits:
        log("No commits found in leaderboard_single.json — is the snapshot empty?")
        return 0

    if args.group:
        # Group by commit.
        for commit in commits:
            present = [w for w in SCENARIO_PARAMS if cell_already_present(w, commit)]
            missing = [w for w in SCENARIO_PARAMS if not cell_already_present(w, commit)]
            exists = commit_exists(HUST_REPO, commit)
            on_main = commit_on_main_branch(HUST_REPO, commit) if exists else False
            branch_mark = "" if on_main else " [non-main]"
            total_missing += len(missing)
            print(f"\n[{commit[:9]}]{branch_mark} ({len(present)}/{len(SCENARIO_PARAMS)} present)")
            for workload in sorted(SCENARIO_PARAMS):
                if not exists:
                    status = "NOT-FOUND"
                elif workload in present:
                    status = "skip"
                else:
                    status = "MISSING"
                print(f"  {status:10s}  {workload}")
    else:
        # Group by workload.
        for workload in sorted(SCENARIO_PARAMS):
            present = [c for c in commits if cell_already_present(workload, c)]
            missing = [c for c in commits if not cell_already_present(workload, c)]
            total_missing += len(missing)
            print(f"\n[{workload}] ({len(present)}/{len(commits)} present)")
            for commit in commits:
                already = cell_already_present(workload, commit)
                exists = commit_exists(HUST_REPO, commit)
                on_main = commit_on_main_branch(HUST_REPO, commit) if exists else False
                branch_mark = "" if on_main else " [non-main]"
                if not exists:
                    status = "NOT-FOUND"
                elif already:
                    status = "skip"
                else:
                    status = "MISSING"
                print(f"  {status:10s}  {workload}  {commit[:9]}{branch_mark}")
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


def _resolve_latest_hust_commit() -> str:
    """Resolve the latest vllm-hust origin/main commit.

    Returns the full 40-char SHA of the latest commit on the main branch.
    Falls back to the current HEAD if ``origin/main`` is not reachable.
    """
    out = subprocess.run(
        ["git", "log", "-1", "--format=%H", "origin/main"],
        cwd=HUST_REPO, capture_output=True, text=True, check=False,
    )
    sha = out.stdout.strip()
    if sha and len(sha) == 40:
        return sha
    # Fallback: use current HEAD if origin/main is not accessible.
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=HUST_REPO, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def cmd_run(args: argparse.Namespace) -> int:
    state = load_state()
    save_state(state)  # Persist the captured HEADs up front.
    log("RUN: starting benchmark")

    # Determine NPU device.
    if args.npu_device is not None:
        npu_id = args.npu_device
        log(f"Using user-specified NPU device: {npu_id}")
    else:
        idle = select_idle_npu()
        if idle is None:
            log("ERROR: no idle NPU device found. "
                "All NPUs have HBM usage above 5000 MB. "
                "Use --npu-device to force a specific device.")
            return 1
        npu_id = idle
        log(f"Auto-selected idle NPU device: {npu_id}")

    # Resolve commit, ascend_commit, and workload targets.
    # --commit and --ascend-commit are optional; if omitted, the value
    # "latest" is used, which resolves to origin/main.  --workload is
    # optional: if given, run only that benchmark; if omitted, run all
    # missing workloads for the commit.
    hust_commit_str = args.commit or "latest"
    ascend_commit_str = args.ascend_commit or "latest"

    if hust_commit_str == "latest":
        hust_commit = _resolve_latest_hust_commit()
        log(f"Using latest vllm-hust origin/main commit: {hust_commit[:12]}")
    else:
        hust_commit = _resolve_full_commit(hust_commit_str)
        log(f"Using vllm-hust commit: {hust_commit[:12]}")

    if ascend_commit_str == "latest":
        ascend_commit = _resolve_compatible_ascend_commit(hust_commit)
        log(f"Auto-resolved ascend commit: {ascend_commit[:12]}")
    else:
        ascend_commit = ascend_commit_str
        log(f"Using ascend commit: {ascend_commit[:12]}")

    if args.workload:
        target = {args.workload: [hust_commit]}
    else:
        # Find missing workloads for this commit.
        missing = [w for w in SCENARIO_PARAMS if not cell_already_present(w, hust_commit)]
        if not missing:
            log(f"All workloads already present for commit {hust_commit[:9]}")
            return 0
        log(f"Missing workloads ({len(missing)}): {', '.join(missing)}")
        target = {w: [hust_commit] for w in missing}

    # Pre-validate all commits exist in the repo.
    missing_commits = []
    for workload, commits in target.items():
        for commit in commits:
            if not commit_exists(HUST_REPO, commit):
                missing_commits.append((workload, commit))

    if missing_commits:
        for w, c in missing_commits:
            log(f"WARNING: commit {c[:9]} (workload={w}) not found in vllm-hust repo, skipping")
        for w, c in missing_commits:
            key = f"{w}:{c[:9]}"
            state["cells"][key] = {"status": "done", "skipped": "commit-not-found"}
        save_state(state)

    # Register a signal handler to restore repos on interrupt.
    _original_hust = state.get("hust_head")
    _original_ascend = state.get("ascend_head")

    def _restore_on_exit() -> None:
        if _original_hust and current_head(HUST_REPO) != _original_hust:
            log(f"Restoring vllm-hust to {_original_hust[:12]}")
            subprocess.run(["git", "checkout", "-fq", _original_hust], cwd=HUST_REPO, check=False)
        if _original_ascend and current_head(ASCEND_REPO) != _original_ascend:
            log(f"Restoring vllm-ascend-hust to {_original_ascend[:12]}")
            subprocess.run(["git", "checkout", "-fq", _original_ascend], cwd=ASCEND_REPO, check=False)

    # Clean up NPU processes before exit.
    def _cleanup_npu() -> None:
        """Kill any remaining vllm server processes on the selected NPU."""
        # Use pkill to find and kill python processes running vllm entrypoints.
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.cli"],
            check=False, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["pkill", "-f", "api_server"],
            check=False, stderr=subprocess.DEVNULL,
        )

    try:
        for workload, commits in target.items():
            for commit in commits:
                if commit in [m[1] for m in missing_commits]:
                    continue  # already skipped
                key = f"{workload}:{commit[:9]}:ascend-{ascend_commit[:9]}"
                existing = state["cells"].get(key, {})
                if existing.get("status") == "done" and not args.force:
                    log(f"SKIP {key} (already done)")
                    continue
                if cell_already_present(workload, commit) and not args.force:
                    log(f"SKIP {key} (already in leaderboard)")
                    state["cells"][key] = {"status": "done", "skipped": "already-present"}
                    continue
                log(f"BEGIN {key}")
                result = run_cell(workload, commit, ascend_commit, npu_id=npu_id)
                state["cells"][key] = result
                save_state(state)
                if result["status"] == "failed":
                    if args.fail_fast:
                        log("FAIL-FAST: stopping after first failure")
                        return 1
    finally:
        _cleanup_npu()
        _restore_on_exit()

    log("RUN: done; remember to run `aggregate` and `push`.")
    return 0


def cmd_fill(args: argparse.Namespace) -> int:
    """Fill all missing benchmark cells across all commits from the data file.

    Iterates over every commit in ``leaderboard_single.json``, finds missing
    workloads, and runs them sequentially.  This is a one-click full backfill
    that combines the discovery of ``plan`` with the execution of ``run``.
    """
    state = load_state()
    save_state(state)
    log("FILL: starting full backfill across all commits")

    # Determine NPU device.
    if args.npu_device is not None:
        npu_id = args.npu_device
        log(f"Using user-specified NPU device: {npu_id}")
    else:
        idle = select_idle_npu()
        if idle is None:
            log("ERROR: no idle NPU device found. "
                "All NPUs have HBM usage above 5000 MB. "
                "Use --npu-device to force a specific device.")
            return 1
        npu_id = idle
        log(f"Auto-selected idle NPU device: {npu_id}")

    commits = sorted(_data_file_commits())
    if not commits:
        log("No commits found in leaderboard_single.json — is the snapshot empty?")
        return 0

    # Optionally filter to a single workload.
    filter_workload = args.workload

    _original_hust = state.get("hust_head")
    _original_ascend = state.get("ascend_head")

    def _restore_on_exit() -> None:
        if _original_hust and current_head(HUST_REPO) != _original_hust:
            log(f"Restoring vllm-hust to {_original_hust[:12]}")
            subprocess.run(["git", "checkout", "-fq", _original_hust], cwd=HUST_REPO, check=False)
        if _original_ascend and current_head(ASCEND_REPO) != _original_ascend:
            log(f"Restoring vllm-ascend-hust to {_original_ascend[:12]}")
            subprocess.run(["git", "checkout", "-fq", _original_ascend], cwd=ASCEND_REPO, check=False)

    def _cleanup_npu() -> None:
        subprocess.run(["pkill", "-f", "vllm.entrypoints.cli"], check=False, stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "api_server"], check=False, stderr=subprocess.DEVNULL)

    total_run = 0
    total_failed = 0
    total_skipped = 0

    try:
        for hust_commit in commits:
            if not commit_exists(HUST_REPO, hust_commit):
                log(f"SKIP commit {hust_commit[:9]}: not found in vllm-hust repo")
                continue

            # Determine which workloads to consider for this commit.
            candidates = [filter_workload] if filter_workload else list(SCENARIO_PARAMS)
            missing = [w for w in candidates if not cell_already_present(w, hust_commit)]
            if not missing:
                log(f"SKIP commit {hust_commit[:9]}: all workloads present")
                continue

            ascend_commit = _resolve_compatible_ascend_commit(hust_commit)
            log(f"=== Commit {hust_commit[:9]}: {len(missing)} missing workload(s) ===")

            for workload in missing:
                key = f"{workload}:{hust_commit[:9]}:ascend-{ascend_commit[:9]}"
                existing = state["cells"].get(key, {})
                if existing.get("status") == "done" and not args.force:
                    log(f"SKIP {key} (already done)")
                    total_skipped += 1
                    continue
                if cell_already_present(workload, hust_commit) and not args.force:
                    log(f"SKIP {key} (already in leaderboard)")
                    state["cells"][key] = {"status": "done", "skipped": "already-present"}
                    total_skipped += 1
                    continue

                log(f"BEGIN {key}")
                result = run_cell(workload, hust_commit, ascend_commit, npu_id=npu_id)
                state["cells"][key] = result
                save_state(state)

                if result["status"] == "failed":
                    total_failed += 1
                    if args.fail_fast:
                        log("FAIL-FAST: stopping after first failure")
                        return 1
                else:
                    total_run += 1
    finally:
        _cleanup_npu()
        _restore_on_exit()

    log(f"FILL: done. {total_run} run, {total_failed} failed, {total_skipped} skipped.")
    log("Remember to run `aggregate` and `push`.")
    return 0 if total_failed == 0 else 1


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


def _remove_excluded_submissions() -> list[Path]:
    """Remove submission directories that are in the public exclusion list.

    Returns the list of removed directory paths.
    """
    exclusions_path = REPO_ROOT / "docs" / "leaderboard-exclusions.json"
    if not exclusions_path.is_file():
        return []

    exclusions = load_leaderboard_exclusions(exclusions_path)
    if not exclusions:
        return []

    submissions_dir = REPO_ROOT / "submissions"
    removed: list[Path] = []
    for sub_dir in sorted(submissions_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        artifact_path = sub_dir / "run_leaderboard.json"
        if not artifact_path.is_file():
            continue
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if match_leaderboard_exclusion(payload, exclusions) is not None:
            shutil.rmtree(sub_dir)
            removed.append(sub_dir)
            log(f"removed excluded submission: {sub_dir.name}")
    return removed


def cmd_aggregate(args: argparse.Namespace) -> int:
    # Remove permanently excluded submissions before aggregation.
    removed = _remove_excluded_submissions()
    if removed:
        log(f"Removed {len(removed)} permanently excluded submission(s) "
            "from the aggregation source.")

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

    p_plan = sub.add_parser("plan", help="Show what is missing.")
    p_plan.add_argument("--group", action="store_true",
                        help="Group output by commit instead of by workload.")
    sub.add_parser("status", help="Show progress from the checkpoint.")
    sub.add_parser("aggregate", help="Rebuild leaderboard-data/snapshots/.")
    sub.add_parser("validate", help="Validate all submissions and snapshots (Section 13.2).")
    sub.add_parser("restore", help="Restore original vllm-hust/ascend HEADs.")

    p_push = sub.add_parser("push", help="Stage, commit and push.")
    p_push.add_argument("-m", "--message", help="Commit message.")
    p_push.add_argument("--dry-run", action="store_true")

    p_run = sub.add_parser("run", help="Run benchmark(s) for a commit.")
    p_run.add_argument("--commit",
                       help="vllm-hust commit to benchmark (optional; if omitted, "
                            "uses latest origin/main).")
    p_run.add_argument("--ascend-commit",
                       help="vllm-ascend-hust plugin commit (optional; if omitted, "
                            "auto-resolves to latest origin/main).")
    p_run.add_argument("--workload", choices=list(SCENARIO_PARAMS.keys()),
                       help="Specific workload to run (optional; if omitted, run all "
                            "missing workloads for the commit).")
    p_run.add_argument("--force", action="store_true",
                       help="Re-run cells already marked done.")
    p_run.add_argument("--fail-fast", action="store_true",
                       help="Stop after the first failed cell.")
    p_run.add_argument("--npu-device", type=int, default=None,
                       help="NPU device index to use (default: auto-select idle NPU via npu-smi).")

    p_fill = sub.add_parser("fill", help="Fill all missing cells across all commits "
                            "(one-click full backfill).")
    p_fill.add_argument("--workload", choices=list(SCENARIO_PARAMS.keys()),
                        help="Specific workload to fill (optional; if omitted, fill all "
                             "missing workloads across all commits).")
    p_fill.add_argument("--force", action="store_true",
                        help="Re-run cells already marked done.")
    p_fill.add_argument("--fail-fast", action="store_true",
                        help="Stop after the first failed cell.")
    p_fill.add_argument("--npu-device", type=int, default=None,
                        help="NPU device index to use (default: auto-select idle NPU via npu-smi).")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    dispatch = {
        "plan": cmd_plan,
        "run": cmd_run,
        "fill": cmd_fill,
        "status": cmd_status,
        "aggregate": cmd_aggregate,
        "validate": cmd_validate,
        "push": cmd_push,
        "restore": cmd_restore,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
