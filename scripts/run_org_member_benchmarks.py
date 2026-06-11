#!/usr/bin/env python3
"""
vLLM-HUST Org Member Benchmark Runner

Fully closed-loop benchmark runner with delta-based attribution for organization members.

Attribution Model:
  - PR commits -> 1 benchmark per PR, attributed to PR
  - Consecutive commits (no PR) by same user -> 1 benchmark per session
  - org member delta = perf(org_group) - perf(previous_upstream_group)
  - Upstream baselines first looked up from existing leaderboard data
  - Upstream results shown on leaderboard as reference baselines

Upstream Baseline Lookup:
  Before running upstream benchmarks, checks for existing results:
    1. Local submissions/ directory in benchmark repo
    2. Leaderboard snapshot (local or GitHub raw content)
    3. If not found -> runs the benchmark

Git Checkout:
  The script checks out each commit before benchmarking, then restores HEAD.

Directory Layout:
  The script auto-detects vllm-hust-benchmark as a sibling directory:
    workspace/
    ├── vllm-hust-benchmark/  <- Auto-detected if sibling exists
    └── hust-tools/
        └── run_org_member_benchmarks.py

Usage:
    python3 run_org_member_benchmarks.py run [OPTIONS]
    python3 run_org_member_benchmarks.py report [OPTIONS]
    python3 run_org_member_benchmarks.py --help

Examples:
    # Run benchmarks (auto-detect sibling vllm-hust-benchmark)
    cd /workspace/hust-tools
    GH_TOKEN=ghp_xxx python3 run_org_member_benchmarks.py run --dry-run

    # Run with upstream baseline commits
    GH_TOKEN=ghp_xxx python3 run_org_member_benchmarks.py run \\
        --upstream-commits abc123,def456

    # Resume with custom settings
    GH_TOKEN=ghp_xxx python3 run_org_member_benchmarks.py run \\
        --resume --model Qwen/Qwen2.5-7B-Instruct

    # Generate reports only
    python3 run_org_member_benchmarks.py report \\
        --checkpoint .benchmarks/checkpoint.json
"""

import argparse
import datetime
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# =============================================================================
# Git Worktree Manager
# =============================================================================

class WorktreeManager:
    """Manages temporary git worktrees for benchmark code isolation.

    Va (stable infrastructure): the main benchmark infra worktree.
    Vb (code under test): a per-run temporary worktree created for each
    commit being benchmarked. Vb is deleted after each run.

    This eliminates the fragility of in-place git checkout + temp file
    fallback for old commits that lack CI script/constraints files.
    """

    def __init__(
        self,
        main_vllm_hust: Path,      # Va: stable vllm-hust worktree
        main_vllm_ascend: Path,    # Va: stable vllm-ascend-hust worktree
        logger_fn=None,
    ):
        self.main_vllm_hust = main_vllm_hust.resolve()
        self.main_vllm_ascend = main_vllm_ascend.resolve()
        self._log = logger_fn or (lambda msg, level=None: print(msg))
        # Worktree base dir: sibling to main repos, sibling parent to Va dirs
        self._base = self.main_vllm_hust.parent / ".benchmarks" / "worktrees"
        self._base.mkdir(parents=True, exist_ok=True)

    def _run(self, cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)

    def _worktree_path(self, sha: str, repo_name: str) -> Path:
        """Path for a named worktree at a given SHA."""
        return self._base / f"{repo_name}--{sha[:12]}"

    def _ensure_worktree(
        self,
        main_repo: Path,
        sha: str,
        repo_name: str,
    ) -> tuple[Path, bool]:
        """Create a worktree for main_repo at SHA if it doesn't exist.

        Fetch strategy:
          1. Check if SHA already exists locally (no fetch needed)
          2. Fetch all refs from origin (covers branches, tags, PR heads)
          3. If still not found, raise RuntimeError for caller fallback

        Returns (worktree_path, was_created).
        """
        wt_path = self._worktree_path(sha, repo_name)
        if wt_path.is_dir():
            try:
                result = self._run(["git", "rev-parse", "--verify", "HEAD"], cwd=wt_path, check=False)
                if result.returncode == 0 and result.stdout.strip() == sha:
                    self._log(f"Reusing existing worktree: {wt_path}")
                    return wt_path, False
                else:
                    self._log(f"Stale worktree at {wt_path} (SHA mismatch), recreating", "warn")
                    self._remove_worktree(wt_path, main_repo)
            except Exception:
                self._remove_worktree(wt_path, main_repo)

        self._log(f"Creating worktree for {repo_name}@{sha[:12]} at {wt_path}")

        # Fast path: SHA already available locally
        if self._sha_exists_locally(main_repo, sha):
            return self._create_worktree_at(main_repo, wt_path, sha)

        # Fetch all refs from origin (branches + tags + PR heads)
        self._log(f"  [Fetch] Fetching all refs for {repo_name}...")
        self._run(["git", "fetch", "origin", "+refs/*:refs/remotes/origin_fetch/*"], cwd=main_repo, check=False)

        if self._sha_exists_locally(main_repo, sha):
            self._log(f"  [Fetch] SHA {sha[:12]} available after fetch")
            return self._create_worktree_at(main_repo, wt_path, sha)

        # SHA not fetchable — signal caller to try fallback
        self._log(f"  [Fetch] SHA {sha[:12]} not available in {repo_name} (may not exist in remote)", "warn")
        raise RuntimeError(f"SHA {sha[:12]} not available in {repo_name}")

    def _sha_exists_locally(self, repo_path: Path, sha: str) -> bool:
        """Check if a commit SHA exists in the local git repo."""
        try:
            result = self._run(
                ["git", "cat-file", "-t", sha],
                cwd=repo_path,
                check=False,
            )
            return result.returncode == 0 and "commit" in result.stdout
        except Exception:
            return False

    def _create_worktree_at(self, source_repo: Path, wt_path: Path, sha: str) -> tuple[Path, bool]:
        """Create a worktree at wt_path for SHA from source_repo."""
        self._run(
            ["git", "worktree", "add", "--force", str(wt_path), sha],
            cwd=source_repo,
            check=True,
        )
        return wt_path, True

    def provision(
        self,
        vllm_hust_sha: str,
        vllm_ascend_sha: str,
        conda_python: str,
        fallback_ascend_sha: str = "",
    ) -> dict[str, Path]:
        """Create Vb worktrees for both repos and install packages.

        If vllm_ascend_sha is not available, falls back to fallback_ascend_sha
        (typically a time-based lookup result).

        Returns dict with 'vllm_hust' and 'vllm_ascend' worktree paths.
        """
        wt_hust, _ = self._ensure_worktree(self.main_vllm_hust, vllm_hust_sha, "vllm-hust")

        try:
            wt_ascend, _ = self._ensure_worktree(self.main_vllm_ascend, vllm_ascend_sha, "vllm-ascend-hust")
        except RuntimeError:
            if fallback_ascend_sha:
                self._log(f"  [Provision] Falling back to time-based SHA: {fallback_ascend_sha[:12]}", "warn")
                wt_ascend, _ = self._ensure_worktree(self.main_vllm_ascend, fallback_ascend_sha, "vllm-ascend-hust")
            else:
                raise

        # Install vllm-hust from Vb in editable mode
        self._install_editable(wt_hust, conda_python, "vllm-hust")
        # Install vllm-ascend-hust from Vb in editable mode
        self._install_editable(wt_ascend, conda_python, "vllm-ascend-hust")

        return {"vllm_hust": wt_hust, "vllm_ascend": wt_ascend}

    def _install_editable(self, worktree: Path, conda_python: str, name: str):
        """Install a package from worktree in editable mode.

        Always runs pip install -e to ensure:
        - Proper package registration
        - Compiled extensions are built for the specific worktree
        - Import resolution is predictable across cross-repo worktrees
        """
        pyproject = worktree / "pyproject.toml"
        if not pyproject.is_file():
            self._log(f"[{name}] No pyproject.toml in {worktree}, skipping editable install", "warn")
            return

        self._log(f"[{name}] Installing editable from {worktree}")
        self._log(f"[{name}] Running: {conda_python} -m pip install --no-deps -e {worktree}")
        try:
            # Stream pip output directly to terminal (no capture)
            result = subprocess.run(
                [conda_python, "-m", "pip", "install", "--no-deps", "-e", str(worktree)],
                cwd=worktree,
                timeout=900,  # 15 min timeout for large packages
            )
            if result.returncode == 0:
                self._log(f"[{name}] Editable install OK", "success")
            else:
                # pip may return non-zero due to warnings (e.g. path string NULL
                # from SWIG, deprecation warnings). Verify the package is actually
                # importable before declaring failure.
                # Map package name to actual Python import name
                # (e.g. vllm-ascend-hust -> vllm_ascend, not vllm_ascend_hust)
                IMPORT_NAME_MAP = {
                    "vllm-hust": "vllm",
                    "vllm-ascend-hust": "vllm_ascend",
                }
                import_name = IMPORT_NAME_MAP.get(name, name.replace("-", "_"))
                try:
                    verify = subprocess.run(
                        [conda_python, "-c", f"import {import_name}"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if verify.returncode == 0:
                        self._log(f"[{name}] Editable install OK (pip exit {result.returncode} ignored, package importable)", "success")
                    else:
                        # stderr likely contains benign warnings; only log as error if
                        # the package is truly not importable.
                        stderr_lines = (result.stderr or "").strip().split("\n")
                        # Also check stdout — cmake error output often goes there
                        stdout_lines = (result.stdout or "").strip().split("\n")
                        # Filter out known benign warning patterns
                        benign = [
                            line for line in stderr_lines
                            if line.strip()
                            and "path string is NULL" not in line
                            and "DeprecationWarning" not in line
                            and not line.startswith("WARNING:")
                        ]
                        if benign:
                            self._log(f"[{name}] Editable install failed (exit {result.returncode})", "error")
                            # Show the first 10 non-benign lines for diagnostics
                            for line in benign[:10]:
                                self._log(f"    {line}", "error")
                            # Also show last 10 lines from stdout (cmake error often there)
                            if stdout_lines:
                                cmake_lines = [l for l in stdout_lines if l.strip()]
                                if cmake_lines:
                                    self._log(f"  cmake output (last {min(10, len(cmake_lines))} lines):", "warn")
                                    for line in cmake_lines[-10:]:
                                        self._log(f"    {line}", "warn")
                        else:
                            self._log(f"[{name}] Editable install OK (pip exit {result.returncode}, only warnings)", "success")
                except Exception:
                    self._log(f"[{name}] Editable install failed (exit {result.returncode})", "error")
        except subprocess.TimeoutExpired:
            self._log(f"[{name}] Editable install timed out (15 min)", "error")
            # Retry once with extended timeout (30 min)
            self._log(f"[{name}] Retrying with extended timeout (30 min)...")
            try:
                result = subprocess.run(
                    [conda_python, "-m", "pip", "install", "--no-deps", "-e", str(worktree)],
                    cwd=worktree,
                    timeout=1800,
                )
                if result.returncode == 0:
                    self._log(f"[{name}] Editable install OK (retry succeeded)", "success")
                else:
                    # Verify importability after retry
                    IMPORT_NAME_MAP = {
                        "vllm-hust": "vllm",
                        "vllm-ascend-hust": "vllm_ascend",
                    }
                    import_name = IMPORT_NAME_MAP.get(name, name.replace("-", "_"))
                    try:
                        verify = subprocess.run(
                            [conda_python, "-c", f"import {import_name}"],
                            capture_output=True, text=True, timeout=10,
                        )
                        if verify.returncode == 0:
                            self._log(f"[{name}] Editable install OK (retry: pip exit {result.returncode} ignored, package importable)", "success")
                        else:
                            self._log(f"[{name}] Editable install failed after retry (exit {result.returncode})", "error")
                    except Exception:
                        self._log(f"[{name}] Editable install failed after retry", "error")
            except subprocess.TimeoutExpired:
                self._log(f"[{name}] Editable install timed out again after retry (30 min)", "error")
                # Verify if package is importable from a previous install
                IMPORT_NAME_MAP = {
                    "vllm-hust": "vllm",
                    "vllm-ascend-hust": "vllm_ascend",
                }
                import_name = IMPORT_NAME_MAP.get(name, name.replace("-", "_"))
                try:
                    verify = subprocess.run(
                        [conda_python, "-c", f"import {import_name}"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if verify.returncode == 0:
                        self._log(
                            f"[{name}] Package still importable from previous install, "
                            f"proceeding with caution",
                            "warn",
                        )
                    else:
                        self._log(
                            f"[{name}] Package NOT importable after double timeout. "
                            f"Benchmark will likely fail.",
                            "error",
                        )
                except Exception:
                    pass
            except Exception as e:
                self._log(f"[{name}] Editable install retry failed: {e}", "error")
        except Exception as e:
            self._log(f"[{name}] Editable install failed: {e}", "error")
            # Non-fatal — conda env package may still work

    def cleanup(self, worktrees: dict[str, Path]):
        """Remove Vb worktrees after benchmark run."""
        for name, wt_path in worktrees.items():
            if wt_path is None or not wt_path.is_dir():
                continue
            repo = self.main_vllm_hust if "vllm-hust" in name else self.main_vllm_ascend
            self._remove_worktree(wt_path, repo)

    def cleanup_stale_worktrees(self) -> int:
        """Scan for and remove residual worktrees from previous crashed runs.

        Called at startup before processing any groups.
        Returns the number of stale worktrees cleaned up.
        """
        if not self._base.is_dir():
            return 0

        stale_dirs = []
        for entry in self._base.iterdir():
            if entry.is_dir() and entry.name != ".git":
                stale_dirs.append(entry)

        # Also check git worktree list for stale references
        for main_repo, label in [(self.main_vllm_hust, "vllm-hust"), (self.main_vllm_ascend, "vllm-ascend-hust")]:
            if main_repo.is_dir():
                try:
                    self._run(["git", "worktree", "prune"], cwd=main_repo, check=False)
                except Exception:
                    pass

        if not stale_dirs:
            return 0

        self._log(f"Found {len(stale_dirs)} stale worktree(s) from previous run(s), cleaning up:")
        for d in stale_dirs:
            self._log(f"  - {d.name}")
            import shutil
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass

        # Prune git worktree references in both repos
        for main_repo in [self.main_vllm_hust, self.main_vllm_ascend]:
            if main_repo.is_dir():
                try:
                    self._run(["git", "worktree", "prune"], cwd=main_repo, check=False)
                except Exception:
                    pass

        self._log(f"Cleaned up {len(stale_dirs)} stale worktree(s)", "success")
        return len(stale_dirs)

    def _remove_worktree(self, wt_path: Path, main_repo: Path):
        """Remove a worktree and its directory."""
        try:
            # Prune the worktree reference from the main repo
            self._run(
                ["git", "worktree", "prune"],
                cwd=main_repo,
                check=True,
            )
            # Also try to remove the worktree branch if it's not main
            self._run(
                ["git", "worktree", "remove", "--force", str(wt_path)],
                cwd=main_repo,
                check=False,  # May fail if worktree is dirty
            )
        except subprocess.CalledProcessError:
            pass

        # Always try to remove the directory
        import shutil
        if wt_path.is_dir():
            try:
                shutil.rmtree(wt_path)
                self._log(f"Removed worktree: {wt_path}")
            except Exception as e:
                self._log(f"Failed to remove worktree dir {wt_path}: {e}", "warn")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for benchmark runner."""
    excluded_members: str = "ShuhaoZhangTony,moonandlife"
    benchmark_scenario: str = "random-online"
    model_name: str = "Qwen/Qwen2.5-14B-Instruct"
    chip_count: int = 1
    since_date: str = "2026-01-01"
    benchmark_branch: str = "main"
    checkpoint_file: str = ".benchmarks/org-member-benchmarks/checkpoint.json"
    attribution_file: str = "contribution-attribution.json"
    html_report_file: str = "contribution-report.html"
    dry_run: bool = False
    resume_mode: bool = False
    include_upstream: bool = True
    upstream_commits: str = ""  # comma-separated upstream commit SHAs for baseline
    baseline_lookup: str = ""  # path to benchmark repo or GitHub raw URL for existing results
    fail_fast: bool = False  # exit on first group failure (for debugging)

    # Paths (set dynamically)
    script_dir: Path = field(default_factory=Path)
    benchmark_repo: Path = field(default_factory=Path)
    vllm_hust_repo: Optional[Path] = None
    vllm_ascend_hust_repo: Optional[Path] = None
    timeline_cache_file: str = ".benchmarks/commit-timeline-cache.json"
    refresh_timeline: bool = False  # Force-refresh cache from GitHub
    timeline_cache_ttl: int = 86400  # Cache TTL in seconds (default 24h)


# =============================================================================
# Model Cache Management (Auto-Detection Logic)
# =============================================================================

def check_huggingface_connectivity(timeout_sec: float = 5.0) -> bool:
    """Check if HuggingFace Hub is reachable."""
    test_hosts = ["huggingface.co", "hf.co"]
    for host in test_hosts:
        try:
            sock = socket.create_connection((host, 443), timeout=timeout_sec)
            sock.close()
            return True
        except (socket.timeout, socket.error, OSError):
            continue
    return False


def get_cached_model_path(model_id: str, conda_python: str) -> Optional[str]:
    """Get cached model path using snapshot_download with local_files_only=True.
    
    Returns the local cache path if model is already downloaded, None otherwise.
    """
    try:
        result = subprocess.run(
            [conda_python, "-c",
             f"import os; from huggingface_hub import snapshot_download; "
             f"print(snapshot_download('{model_id}', local_files_only=True))"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            path = result.stdout.strip()
            if path and os.path.exists(path):
                # Verify essential files exist (config.json is required)
                if os.path.exists(os.path.join(path, "config.json")):
                    return path
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


def has_required_model_artifacts(model_path: str) -> bool:
    """Check if a local model path has required artifacts (config.json + tokenizer)."""
    if not os.path.exists(model_path):
        return False
    
    # Check for config.json (required)
    if not os.path.exists(os.path.join(model_path, "config.json")):
        return False
    
    # Check for tokenizer files (safetensors or pytorch_model.bin or config.json for vocab)
    has_weights = (
        os.path.exists(os.path.join(model_path, "model.safetensors")) or
        os.path.exists(os.path.join(model_path, "pytorch_model.bin")) or
        any(f.endswith(".bin") for f in os.listdir(model_path)) or
        any(f.endswith(".safetensors") for f in os.listdir(model_path))
    )
    has_tokenizer = (
        os.path.exists(os.path.join(model_path, "tokenizer.json")) or
        os.path.exists(os.path.join(model_path, "tokenizer_config.json")) or
        os.path.exists(os.path.join(model_path, "spiece.model"))
    )
    
    # Return true if weights exist (tokenizer might be optional in some cases)
    return has_weights or has_tokenizer


def download_model_from_hub(
    model_id: str,
    target_dir: str,
    conda_python: str,
    logger_fn=None,
    hf_endpoint: str = "https://hf-mirror.com"
) -> bool:
    """Download model from HuggingFace Hub using 'hf download' command.
    
    Args:
        model_id: HuggingFace model ID (e.g., 'Qwen/Qwen2.5-14B-Instruct')
        target_dir: Local directory to download model to
        conda_python: Path to Python interpreter with huggingface_hub installed
        logger_fn: Optional logging function
        hf_endpoint: HF_ENDPOINT mirror URL (default: hf-mirror.com)
    
    Returns:
        True if download succeeded, False otherwise
    """
    log = logger_fn if logger_fn else (lambda msg, level=None: print(msg))
    
    log(f"  [Model] Downloading {model_id} to {target_dir}...")
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Use 'hf download' command (replaces deprecated 'huggingface-cli download')
        # Note: huggingface_hub.commands.huggingface_cli was removed in huggingface_hub >= 1.0.0
        # Use the standalone 'hf' CLI instead, which is available in the conda environment.
        env = os.environ.copy()
        env["HF_ENDPOINT"] = hf_endpoint

        result = subprocess.run(
            ["hf", "download", model_id,
             "--local-dir", target_dir],
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for large models
        )
        
        if result.returncode == 0:
            log(f"  [Model] Download completed successfully", "success")
            return True
        else:
            log(f"  [Model] Download failed: {result.stderr[:500]}", "error")
            return False
            
    except subprocess.TimeoutExpired:
        log(f"  [Model] Download timed out (>1 hour)", "error")
        return False
    except Exception as e:
        log(f"  [Model] Download error: {e}", "error")
        return False


def resolve_effective_model(
    model_id: str,
    explicit_model_path: Optional[str],
    conda_python: str,
    logger_fn=None,
    auto_download: bool = True,
    default_cache_dir: str = "/data/shared_models",
    hf_endpoint: str = "https://hf-mirror.com",
) -> tuple[str, str, bool]:
    """Resolve the effective model to use, with auto-detection and download logic.
    
    Returns: (effective_model_path_or_id, mode_description, is_offline)
    
    Modes:
    - "explicit_local": Uses explicitly provided local path
    - "cached_local": Uses HuggingFace cache if model is already downloaded
    - "auto_downloaded": Model was downloaded to default cache directory
    - "online_download": Will download from Hub (auto_download=False)
    - "offline_unavailable": No cache and network unavailable
    """
    log = logger_fn if logger_fn else (lambda msg, level=None: print(msg))
    
    # Mode 0: model_id itself is a local path (absolute or relative)
    # This handles the common case where --model is given as a full path
    # like /data/shared_models/Qwen--Qwen2.5-14B-Instruct
    if model_id and os.path.isabs(model_id) and os.path.isdir(model_id):
        if has_required_model_artifacts(model_id):
            log(f"  [Model] Using local path from model_id: {model_id}", "info")
            return model_id, "cached_local", True
        else:
            log(f"  [Model] model_id is a local dir but missing artifacts: {model_id}", "warn")
    
    # Mode 1: Use explicitly provided local path
    if explicit_model_path and os.path.exists(explicit_model_path):
        if has_required_model_artifacts(explicit_model_path):
            log(f"  [Model] Using explicit local path: {explicit_model_path}", "info")
            return explicit_model_path, "explicit_local", False
        else:
            log(f"  [Model] Explicit path exists but missing artifacts: {explicit_model_path}", "warn")
    
    # If model_id looks like a local path (absolute), don't treat it as HF repo ID
    if model_id and os.path.isabs(model_id):
        if os.path.isdir(model_id):
            log(f"  [Model] ERROR: Local path '{model_id}' exists but is missing model artifacts", "error")
            log(f"  [Model] Expected files: config.json, tokenizer.json or tokenizer.model", "error")
        else:
            log(f"  [Model] ERROR: Local path '{model_id}' does not exist", "error")
        return model_id, "offline_unavailable", True
    
    # Mode 2: Check HuggingFace cache for existing model
    cached_path = get_cached_model_path(model_id, conda_python)
    if cached_path and has_required_model_artifacts(cached_path):
        log(f"  [Model] Using cached local model: {cached_path}", "info")
        return cached_path, "cached_local", True
    
    # Mode 3: Check default cache directory
    default_model_path = os.path.join(default_cache_dir, model_id.replace("/", "--"))
    if os.path.exists(default_model_path) and has_required_model_artifacts(default_model_path):
        log(f"  [Model] Using pre-downloaded model in {default_model_path}", "info")
        return default_model_path, "cached_local", True
    
    # Mode 4: Network connectivity check
    if check_huggingface_connectivity():
        if auto_download:
            # Auto-download to default cache directory
            if download_model_from_hub(model_id, default_model_path, conda_python, logger_fn, hf_endpoint):
                if os.path.exists(default_model_path) and has_required_model_artifacts(default_model_path):
                    log(f"  [Model] Using auto-downloaded model: {default_model_path}", "info")
                    return default_model_path, "auto_downloaded", False
        
        log(f"  [Model] Model not cached, will download on demand: {model_id}", "info")
        return model_id, "online_download", False
    
    # Mode 5: Offline fallback - try cache again without network check
    if cached_path:
        log(f"  [Model] Network unavailable, using cached model: {cached_path}", "warn")
        return cached_path, "cached_local", True
    
    # Check default cache again in case it exists locally
    if os.path.exists(default_model_path) and has_required_model_artifacts(default_model_path):
        log(f"  [Model] Network unavailable, using pre-downloaded model: {default_model_path}", "warn")
        return default_model_path, "cached_local", True
    
    # Mode 6: No cache, no network - fail with clear message
    log(f"  [Model] ERROR: Model '{model_id}' not in cache and HuggingFace Hub unreachable", "error")
    log(f"  [Model] Please either:", "error")
    log(f"    1. Pre-download model: HF_ENDPOINT={hf_endpoint} huggingface-cli download {model_id}", "error")
    log(f"    2. Set HF_HUB_OFFLINE=1 and ensure model is in cache", "error")
    return model_id, "offline_unavailable", True


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class CommitInfo:
    """Information about a commit."""
    sha: str
    date: str
    user: str
    pr_number: Optional[int] = None
    message: str = ""


@dataclass
class CommitGroup:
    """A group of commits to benchmark together."""
    group_id: str
    group_type: str  # "pr", "session"
    commits: list[CommitInfo]
    is_org_member: bool
    pr_number: Optional[int] = None
    pr_title: str = ""

    @property
    def latest_commit(self) -> str:
        return self.commits[-1].sha if self.commits else ""

    @property
    def latest_date(self) -> str:
        return self.commits[-1].date if self.commits else ""

    @property
    def submitter(self) -> str:
        return self.commits[0].user if self.commits else ""


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    throughput_tps: float
    ttft_ms: float
    success: bool
    output_dir: str = ""


@dataclass
class AttributionEntry:
    """Attribution entry for an org member contribution."""
    org_member: str
    group_id: str
    group_type: str
    pr_number: Optional[int]
    commit_count: int
    latest_commit: str
    latest_commit_date: str
    upstream_baseline_commit: str
    upstream_throughput_tps: float
    org_throughput_tps: float
    delta_tps: float
    improvement_percent: float
    ttft_improvement_percent: float
    attribution_timestamp: str = ""


@dataclass
class TimelineCache:
    """Cached commit timeline to avoid repeated GitHub API calls.
    
    Reduces GitHub API usage and speeds up subsequent runs. Default TTL: 24h.
    Can be force-refreshed with --refresh-timeline flag.
    """
    org_members: list[str]
    all_commits: list[dict]  # [{date, sha, user, is_org_member}]
    pr_cache: list[dict]     # [{sha, number, title, user}]
    cached_at: str           # ISO timestamp when cached
    cache_version: int = 1   # Schema version for forward compat

    def is_stale(self, ttl_seconds: int = 86400) -> bool:
        """Check if cache is older than TTL seconds."""
        import datetime
        try:
            cached = datetime.datetime.fromisoformat(self.cached_at.replace("Z", "+00:00"))
            now = datetime.datetime.now(datetime.timezone.utc)
            return (now - cached).total_seconds() > ttl_seconds
        except Exception:
            return True  # Treat unparseable timestamps as stale

    def to_dict(self) -> dict:
        return {
            "org_members": self.org_members,
            "all_commits": self.all_commits,
            "pr_cache": self.pr_cache,
            "cached_at": self.cached_at,
            "cache_version": self.cache_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TimelineCache":
        return cls(
            org_members=data.get("org_members", []),
            all_commits=data.get("all_commits", []),
            pr_cache=data.get("pr_cache", []),
            cached_at=data.get("cached_at", ""),
            cache_version=data.get("cache_version", 1),
        )


@dataclass
class Checkpoint:
    """Checkpoint state for resume capability."""
    groups_processed: dict[str, dict] = field(default_factory=dict)
    benchmarks_completed: dict[str, int] = field(default_factory=lambda: {"success": 0, "failed": 0})
    upstream_baselines: dict[str, dict] = field(default_factory=dict)
    attribution: list[dict] = field(default_factory=list)
    last_updated: str = ""

    @classmethod
    def from_file(cls, path: Path) -> "Checkpoint":
        """Load checkpoint from JSON file."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(
                groups_processed=data.get("groups_processed", {}),
                benchmarks_completed=data.get("benchmarks_completed", {"success": 0, "failed": 0}),
                upstream_baselines=data.get("upstream_baselines", {}),
                attribution=data.get("attribution", []),
                last_updated=data.get("last_updated", "")
            )
        return cls()

    def to_file(self, path: Path) -> None:
        """Save checkpoint to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# =============================================================================
# GitHub API Client
# =============================================================================

class GitHubClient:
    """GitHub API client."""

    def __init__(self, token: str):
        self.token = token
        self.base_url = "https://api.github.com"

    def _request(self, endpoint: str) -> Any:
        """Make authenticated API request."""
        cmd = [
            "curl", "-sS",
            "-H", f"Authorization: token {self.token}",
            "-H", "Accept: application/vnd.github.v3+json",
            f"{self.base_url}{endpoint}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)

    def get_org_members(self, org: str = "vLLM-HUST") -> list[str]:
        """Get organization members."""
        members = self._request(f"/orgs/{org}/members?per_page=100")
        return [m["login"] for m in members]

    def get_user_commits(self, repo: str, user: str, since: str) -> list[CommitInfo]:
        """Get commits by a specific user since a date."""
        commits = self._request(
            f"/repos/{repo}/commits?author={user}&since={since}T00:00:00Z&per_page=100"
        )
        result = []
        for c in commits:
            result.append(CommitInfo(
                sha=c["sha"],
                date=c["commit"]["author"]["date"],
                user=user,
                pr_number=None,
                message=c["commit"]["message"].split("\n")[0].strip()
            ))
        return result

    def get_merged_prs(self, repo: str) -> dict[int, str]:
        """Get merged PRs and their merge commit SHAs."""
        pr_map = {}
        page = 1
        while True:
            prs = self._request(f"/repos/{repo}/pulls?state=closed&per_page=100&page={page}")
            if not prs:
                break
            for pr in prs:
                if pr.get("merged_at"):
                    pr_map[pr["number"]] = pr["merge_commit_sha"]
            if len(prs) < 100:
                break
            page += 1
        return pr_map

    def get_commit_pr(self, repo: str, sha: str) -> Optional[dict]:
        """Get the PR associated with a commit SHA (if any).

        Returns dict with 'number', 'title', 'user', 'merge_commit_sha' or None if not a PR commit.
        """
        try:
            prs = self._request(f"/repos/{repo}/commits/{sha}/pulls")
            if prs and len(prs) > 0:
                pr = prs[0]
                return {
                    "number": pr["number"],
                    "title": pr.get("title", ""),
                    "user": pr.get("user", {}).get("login", ""),
                    "merge_commit_sha": pr.get("merge_commit_sha", ""),
                }
        except Exception:
            pass
        return None

    def get_pr_merge_sha(self, repo: str, pr_number: int) -> Optional[str]:
        """Get the merge commit SHA for a specific PR in a repo."""
        try:
            pr = self._request(f"/repos/{repo}/pulls/{pr_number}")
            return pr.get("merge_commit_sha", "") or None
        except Exception:
            return None

    def get_all_commits(self, repo: str, since: str, branch: str = "main") -> list[CommitInfo]:
        """Get all commits on a branch since a date (not filtered by author)."""
        commits = self._request(
            f"/repos/{repo}/commits?sha={branch}&since={since}T00:00:00Z&per_page=100"
        )
        result = []
        for c in commits:
            # Determine author login
            author = ""
            if c.get("author"):
                author = c["author"]["login"]
            elif c.get("commit", {}).get("author", {}).get("name"):
                author = c["commit"]["author"]["name"]
            result.append(CommitInfo(
                sha=c["sha"],
                date=c["commit"]["author"]["date"],
                user=author,
                message=c["commit"]["message"].split("\n")[0].strip()
            ))
        return result


# =============================================================================
# Upstream Baseline Lookup
# =============================================================================

class UpstreamBaselineLookup:
    """Looks up existing upstream benchmark results from leaderboard data.

    Lookup order:
      1. Local benchmark repo leaderboard-data/snapshots/leaderboard_single.json
      2. GitHub raw content from benchmark repo
      3. Not found -> need to run benchmark
    """

    GITHUB_RAW_URL = (
        "https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark"
        "/main/leaderboard-data/snapshots/leaderboard_single.json"
    )

    def __init__(self, benchmark_repo: Optional[Path] = None, github_token: str = ""):
        self.benchmark_repo = benchmark_repo
        self.github_token = github_token
        self._cache: Optional[list[dict]] = None

    def _load_local(self) -> Optional[list[dict]]:
        """Try to load from local benchmark repo snapshots."""
        if not self.benchmark_repo:
            return None
        snapshot_path = self.benchmark_repo / "leaderboard-data" / "snapshots" / "leaderboard_single.json"
        if snapshot_path.exists():
            try:
                with open(snapshot_path) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return data
            except Exception:
                pass
        return None

    def _load_github(self) -> Optional[list[dict]]:
        """Try to fetch from GitHub raw content."""
        try:
            req = urllib.request.Request(self.GITHUB_RAW_URL)
            if self.github_token:
                req.add_header("Authorization", f"token {self.github_token}")
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return None

    def _get_entries(self) -> list[dict]:
        """Get leaderboard entries (with caching)."""
        if self._cache is not None:
            return self._cache

        # Try local first, then GitHub
        entries = self._load_local()
        if entries:
            self._cache = entries
            return entries

        entries = self._load_github()
        if entries:
            self._cache = entries
            return entries

        self._cache = []
        return []

    def lookup(self, commit_sha: str) -> Optional[dict]:
        """Look up benchmark results for a specific commit SHA.

        Returns dict with 'throughput_tps', 'ttft_ms', 'source' or None.
        """
        entries = self._get_entries()
        sha_short = commit_sha[:12]
        sha_full = commit_sha

        for entry in entries:
            # Check if entry's engine_version or commit info matches
            # The leaderboard entries may have commit SHA in various fields
            env = entry.get("environment", {})
            versions = entry.get("versions", {})

            # Check multiple possible locations for commit SHA
            for field_name in ("commit_sha", "git_sha", "commit", "sha"):
                val = env.get(field_name, "") or versions.get(field_name, "")
                if val and (val == sha_full or val.startswith(sha_short) or
                            sha_full.startswith(val[:12] if len(val) >= 12 else val)):
                    metrics = entry.get("metrics", {})
                    return {
                        "throughput_tps": metrics.get("throughput_tps", 0),
                        "ttft_ms": metrics.get("ttft_ms", 0),
                        "source": "leaderboard_lookup",
                        "entry_id": entry.get("entry_id", ""),
                    }

            # Also check entry_id or run_id patterns that might contain SHA
            entry_id = entry.get("entry_id", "")
            if sha_short in entry_id:
                metrics = entry.get("metrics", {})
                return {
                    "throughput_tps": metrics.get("throughput_tps", 0),
                    "ttft_ms": metrics.get("ttft_ms", 0),
                    "source": "leaderboard_lookup",
                    "entry_id": entry_id,
                }

        return None

    def lookup_by_submission_dir(self, benchmark_repo: Path, run_id: str) -> Optional[dict]:
        """Look up results from local submissions directory."""
        submission_dir = benchmark_repo / "submissions" / run_id
        leaderboard_file = submission_dir / "run_leaderboard.json"
        if leaderboard_file.exists():
            try:
                with open(leaderboard_file) as f:
                    data = json.load(f)
                metrics = data.get("metrics", {})
                return {
                    "throughput_tps": metrics.get("throughput_tps", 0),
                    "ttft_ms": metrics.get("ttft_ms", 0),
                    "source": "local_submission",
                    "entry_id": run_id,
                }
            except Exception:
                pass
        return None


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Runs benchmarks for org member commits."""

    def __init__(self, config: Config, github_token: str, log_file: str = ""):
        self.config = config
        self.github = GitHubClient(github_token)
        self.checkpoint = Checkpoint.from_file(Path(config.checkpoint_file))
        self._cann_env = self._detect_cann_env()
        self._baseline_lookup = UpstreamBaselineLookup(
            benchmark_repo=config.benchmark_repo if config.baseline_lookup != "no-lookup" else None,
            github_token=github_token,
        )
        self._original_head: str = ""  # for git restore
        self._original_ascend_head: str = ""  # for legacy mode ascend restore
        self._log_fh: Optional[Any] = None
        self._offline_mode: bool = False
        self._effective_model: str = config.model_name
        self._model_resolution_mode: str = ""
        self._timeline_ttl: int = 86400  # Cache TTL in seconds

        # Worktree manager — Va is the stable infrastructure worktree
        self._va_vllm_hust = (
            (config.vllm_hust_repo or benchmark_repo).resolve()
            if config.vllm_hust_repo
            else None
        )
        self._va_vllm_ascend = config.vllm_ascend_hust_repo.resolve() if config.vllm_ascend_hust_repo else None

        if self._va_vllm_hust and self._va_vllm_ascend:
            self._worktree_mgr = WorktreeManager(
                main_vllm_hust=self._va_vllm_hust,
                main_vllm_ascend=self._va_vllm_ascend,
                logger_fn=lambda msg, level=None: self._log(msg, level or "info"),
            )
            self._log("Worktree-based benchmark mode enabled (Va/Vb isolation)")
        else:
            self._worktree_mgr = None
            if config.vllm_hust_repo:
                self._log("vllm_ascend_hust_repo not configured, falling back to in-place checkout", "warn")
            else:
                self._log("vllm_hust_repo not configured, falling back to in-place checkout", "warn")

        # Timeline cache TTL (from CLI, default 24h)
        self._timeline_ttl = getattr(config, "timeline_cache_ttl", 86400)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(log_path, "w", buffering=1)  # line-buffered
            self._log(f"Logging to: {log_path.resolve()}")
        
        # Auto-detect model and set up offline mode if needed
        self._resolve_model()

    @staticmethod
    def _detect_cann_env() -> dict[str, str]:
        """Detect CANN toolkit environment by sourcing set_env.sh.

        Captures the FULL environment after sourcing set_env.sh so that
        subprocesses receive every variable the CANN toolchain needs
        (PYTHONPATH, LD_LIBRARY_PATH, ASCEND_HOME_PATH, ASCEND_RT_VISIBLE_DEVICES,
        PATH, and dozens of ASCEND_*/ATB_*/AICPU_* variables).
        """
        set_env_candidates = [
            "/usr/local/Ascend/ascend-toolkit/set_env.sh",
            "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh",
        ]
        set_env_path = ""
        for candidate in set_env_candidates:
            if os.path.isfile(candidate):
                set_env_path = candidate
                break

        if not set_env_path:
            return {}

        try:
            # Source set_env.sh in a subshell, then dump the full environment
            # using `env -0` (NUL-delimited for safe parsing).
            script = (
                f"source {shlex.quote(set_env_path)} 2>/dev/null && "
                # Also source ATB env if available
                "if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then "
                "  source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 2>/dev/null; "
                "fi && "
                "echo __CANN_ENV_MARKER__ && "
                "env -0"
            )
            result = subprocess.run(
                ["bash", "-c", script],
                capture_output=True, text=True, timeout=15,
            )
            cann_env = {}
            in_section = False
            for chunk in result.stdout.split("\0"):
                if "__CANN_ENV_MARKER__" in chunk:
                    in_section = True
                    # There might be a partial line after the marker
                    _, _, remainder = chunk.partition("__CANN_ENV_MARKER__")
                    if "=" in remainder:
                        key, _, value = remainder.partition("=")
                        if key and value:
                            cann_env[key] = value
                    continue
                if in_section and "=" in chunk:
                    key, _, value = chunk.partition("=")
                    if key and value:
                        cann_env[key] = value
            if cann_env:
                print(f"[CANN] Detected {len(cann_env)} env vars from {set_env_path}")
            return cann_env
        except Exception as exc:
            print(f"[CANN] Warning: failed to detect CANN env: {exc}")
            return {}

    def _log(self, msg: str, style: str = "info"):
        """Print log message with style (tees to log file if open)."""
        colors = {"info": "\033[94m", "success": "\033[92m", "warn": "\033[93m",
                  "error": "\033[91m", "section": "\033[95m"}
        color = colors.get(style, "")
        line = f"[{style.upper():8s}] {msg}"
        print(f"{color}{line}\033[0m")
        if self._log_fh:
            self._log_fh.write(line + "\n")

    def _print_section(self, title: str):
        """Print section header."""
        width = 60
        pad = (width - len(title) - 2) // 2
        header = f"\n{'=' * width}\n{' ' * pad}#{title}{' ' * (width - pad - len(title) - 1)}#\n{'=' * width}\n"
        print(f"\033[95m{header}\033[0m")
        if self._log_fh:
            # Write without color codes
            self._log_fh.write(header)

    def _tee_print(self, line: str):
        """Print a line to terminal and log file (for benchmark streaming output)."""
        print(f"  | {line}", flush=True)
        if self._log_fh:
            self._log_fh.write(f"  | {line}\n")

    def _resolve_model(self):
        """Resolve effective model with auto-detection and auto-download logic.
        
        This checks for cached models and network availability, then:
        - Uses cached local model if available
        - Auto-downloads to default cache directory if not cached and network available
        - Falls back to HF_HUB_OFFLINE if network is unavailable
        - Sets appropriate environment variables for the benchmark CLI
        """
        # Determine conda python path
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        conda_python = f"{conda_prefix}/bin/python" if conda_prefix else sys.executable
        if not os.path.exists(conda_python):
            conda_python = sys.executable
        
        # Get default cache directory and HF endpoint from environment
        default_cache_dir = os.environ.get("VLLM_HUST_MODEL_CACHE_DIR", "/data/shared_models")
        hf_endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
        
        # Check for explicit model path from environment
        explicit_path = os.environ.get("VLLM_HUST_MODEL_PATH", "")
        
        # Resolve model with auto-download enabled
        effective_model, mode, is_offline = resolve_effective_model(
            model_id=self.config.model_name,
            explicit_model_path=explicit_path if explicit_path else None,
            conda_python=conda_python,
            logger_fn=lambda msg, level=None: self._log(msg, level or "info"),
            auto_download=True,  # Enable auto-download
            default_cache_dir=default_cache_dir,
            hf_endpoint=hf_endpoint,
        )
        
        self._effective_model = effective_model
        self._model_resolution_mode = mode
        self._offline_mode = is_offline
        
        # Log resolution summary
        mode_descriptions = {
            "explicit_local": "using explicit local model path",
            "cached_local": "using local cache (offline mode)",
            "auto_downloaded": "using auto-downloaded model",
            "online_download": "will download on demand",
            "offline_unavailable": "offline mode but model not cached",
        }
        desc = mode_descriptions.get(mode, mode)
        self._log(f"Model resolution: {desc}", "section")
        
        # Set offline environment variables if needed
        if is_offline and mode not in ("online_download", "auto_downloaded"):
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self._log("Offline mode enabled (HF_HUB_OFFLINE=1)", "info")

    def _git_save_head(self):
        """Save current HEAD for later restore.

        NOTE: This is only used when worktree mode is DISABLED (legacy fallback).
        When worktree mode is enabled, HEAD is never moved in Va.
        Also saves vllm-ascend-hust HEAD for version coordination.
        """
        if self.config.dry_run:
            return
        # Use vllm_hust_repo for git operations if available, else benchmark_repo
        git_repo = self.config.vllm_hust_repo or self.config.benchmark_repo
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=git_repo,
                capture_output=True, text=True, check=True,
            )
            self._original_head = result.stdout.strip()
            self._log(f"Saved HEAD: {self._original_head[:12]}")
        except Exception as e:
            self._log(f"Failed to save HEAD: {e}", "warn")

        # Also save vllm-ascend-hust HEAD for legacy mode coordination
        ascend_repo = self.config.vllm_ascend_hust_repo
        if ascend_repo and ascend_repo.is_dir():
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ascend_repo,
                    capture_output=True, text=True, check=True,
                )
                self._original_ascend_head = result.stdout.strip()
                self._log(f"Saved Ascend HEAD: {self._original_ascend_head[:12]}")
            except Exception as e:
                self._log(f"Failed to save Ascend HEAD: {e}", "warn")

    def _save_ci_script_copy(self):
        """Save CI script and constraints from Va infrastructure dirs.

        NOTE: This is only used when worktree mode is DISABLED. With worktree mode,
        VLLM_HUST_REPO points to Va (stable) and BENCH_CONSTRAINTS_FILE is set
        from Va's data dir — no extraction or temp file needed.
        """
        """"Save the CI script to a stable temp location before checking out old commits.

        Old commits may not have the CI script file, so we extract it from the
        current HEAD (main branch) and write it to a temp file that persists
        across all benchmark runs.
        """
        self._ci_script_stable_path = None
        git_repo = self.config.vllm_hust_repo or self.config.benchmark_repo
        ci_script_rel = ".github/workflows/scripts/run_ascend_benchmark_ci.sh"


        # Prefer a pre-patched CI script in the workspace (with --gpu-memory-utilization fix)
        workspace_ci = Path("/workspace/benchmark_backfill/_ci_scripts/run_ascend_benchmark_ci.sh")
        if workspace_ci.is_file():
            import tempfile, shutil
            tmp_dir = Path(tempfile.gettempdir()) / "vllm-hust-benchmark-ci"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            stable_path = tmp_dir / "run_ascend_benchmark_ci.sh"
            shutil.copy2(str(workspace_ci), str(stable_path))
            stable_path.chmod(0o755)
            self._ci_script_stable_path = stable_path
            self._log(f"CI script (workspace patched) saved to stable path: {stable_path}")
            return

        ci_script_abs = Path(git_repo) / ci_script_rel

        if ci_script_abs.is_file():
            # Copy to a stable temp location
            import tempfile
            tmp_dir = Path(tempfile.gettempdir()) / "vllm-hust-benchmark-ci"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            stable_path = tmp_dir / "run_ascend_benchmark_ci.sh"
            import shutil
            shutil.copy2(str(ci_script_abs), str(stable_path))
            stable_path.chmod(0o755)
            self._ci_script_stable_path = stable_path
            self._log(f"CI script saved to stable path: {stable_path}")
        else:
            # Try to extract from git (main branch) — but only if the file exists
            try:
                # Verify the file exists in origin/main before attempting git show
                check = subprocess.run(
                    ["git", "ls-tree", "origin/main", ci_script_rel],
                    cwd=git_repo, capture_output=True, text=True,
                )
                if check.returncode != 0 or not check.stdout.strip():
                    self._log(f"CI script not found in origin/main ({ci_script_rel}), skipping", "warn")
                else:
                    result = subprocess.run(
                        ["git", "show", f"origin/main:{ci_script_rel}"],
                        cwd=git_repo, capture_output=True, text=True, check=True,
                    )
                    import tempfile
                    tmp_dir = Path(tempfile.gettempdir()) / "vllm-hust-benchmark-ci"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    stable_path = tmp_dir / "run_ascend_benchmark_ci.sh"
                    stable_path.write_text(result.stdout)
                    stable_path.chmod(0o755)
                    self._ci_script_stable_path = stable_path
                    self._log(f"CI script extracted from origin/main to: {stable_path}")
            except subprocess.CalledProcessError as e:
                self._log(f"Failed to save CI script copy: {e.stderr.strip()}", "warn")
            except Exception as e:
                self._log(f"Failed to save CI script copy: {e}", "warn")

        # Also save constraints file (same problem: old commits lack it)
        self._constraints_file_stable_path = None
        constraints_rel = ".github/workflows/data/random-online-ci-constraints.json"
        constraints_abs = Path(git_repo) / constraints_rel

        if constraints_abs.is_file():
            import tempfile, shutil
            tmp_dir = Path(tempfile.gettempdir()) / "vllm-hust-benchmark-ci"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            stable_constraints = tmp_dir / "random-online-ci-constraints.json"
            shutil.copy2(str(constraints_abs), str(stable_constraints))
            self._constraints_file_stable_path = stable_constraints
            self._log(f"Constraints file saved to: {stable_constraints}")
        else:
            # Try to extract from git (main branch) — but only if the file exists
            try:
                # Verify the file exists in origin/main before attempting git show
                check = subprocess.run(
                    ["git", "ls-tree", "origin/main", constraints_rel],
                    cwd=git_repo, capture_output=True, text=True,
                )
                if check.returncode != 0 or not check.stdout.strip():
                    self._log(f"Constraints file not found in origin/main ({constraints_rel}), skipping", "warn")
                else:
                    result = subprocess.run(
                        ["git", "show", f"origin/main:{constraints_rel}"],
                        cwd=git_repo, capture_output=True, text=True, check=True,
                    )
                    import tempfile
                    tmp_dir = Path(tempfile.gettempdir()) / "vllm-hust-benchmark-ci"
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    stable_constraints = tmp_dir / "random-online-ci-constraints.json"
                    stable_constraints.write_text(result.stdout)
                    self._constraints_file_stable_path = stable_constraints
                    self._log(f"Constraints file extracted from origin/main to: {stable_constraints}")
            except subprocess.CalledProcessError as e:
                self._log(f"Failed to save constraints file: {e.stderr.strip()}", "warn")
            except Exception as e:
                self._log(f"Failed to save constraints file: {e}", "warn")

    def _git_checkout_fallback(self, sha: str) -> bool:
        """Legacy in-place git checkout (fallback when worktree mode is disabled)."""
        if self.config.dry_run:
            self._log(f"[DRY RUN] Would checkout {sha[:12]}")
            return True
        # Use vllm_hust_repo for git operations if available, else benchmark_repo
        git_repo = self.config.vllm_hust_repo or self.config.benchmark_repo
        try:
            # Fetch latest from remote to ensure commit is available
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=git_repo, capture_output=True,
            )
            # Stash any local changes first
            subprocess.run(
                ["git", "stash", "--include-untracked", "-m", "org-benchmark-auto"],
                cwd=git_repo, capture_output=True,
            )
            result = subprocess.run(
                ["git", "checkout", sha],
                cwd=git_repo, capture_output=True, text=True,
            )
            if result.returncode != 0:
                self._log(f"Git checkout {sha[:12]} failed: {result.stderr.strip()}", "error")
                return False
            self._log(f"Checked out {sha[:12]}")
            return True
        except Exception as e:
            self._log(f"Git checkout failed: {e}", "error")
            return False

    def _git_checkout_ascend(self, ascend_repo: Path, sha: str) -> bool:
        """Checkout vllm-ascend-hust to a specific commit (legacy mode)."""
        if self.config.dry_run:
            self._log(f"  [Ascend] Would checkout {sha[:12]}")
            return True
        try:
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=ascend_repo, capture_output=True, timeout=30,
            )
            result = subprocess.run(
                ["git", "checkout", sha],
                cwd=ascend_repo, capture_output=True, text=True,
            )
            if result.returncode != 0:
                self._log(f"  [Ascend] Checkout {sha[:12]} failed: {result.stderr.strip()}", "error")
                return False
            self._log(f"  [Ascend] Checked out {sha[:12]}")
            return True
        except Exception as e:
            self._log(f"  [Ascend] Checkout failed: {e}", "error")
            return False

    def _git_restore_head(self):
        """Restore original HEAD after all benchmarks."""
        if self.config.dry_run or not self._original_head:
            return
        # Use vllm_hust_repo for git operations if available, else benchmark_repo
        git_repo = self.config.vllm_hust_repo or self.config.benchmark_repo
        try:
            subprocess.run(
                ["git", "checkout", self._original_head],
                cwd=git_repo, capture_output=True,
            )
            # Pop stash if we created one
            subprocess.run(
                ["git", "stash", "pop"],
                cwd=git_repo, capture_output=True,
            )
            self._log(f"Restored HEAD to {self._original_head[:12]}")
        except Exception as e:
            self._log(f"Failed to restore HEAD: {e}", "warn")

        # Also restore vllm-ascend-hust HEAD
        ascend_repo = self.config.vllm_ascend_hust_repo
        if ascend_repo and ascend_repo.is_dir() and self._original_ascend_head:
            try:
                subprocess.run(
                    ["git", "checkout", self._original_ascend_head],
                    cwd=ascend_repo, capture_output=True,
                )
                self._log(f"Restored Ascend HEAD to {self._original_ascend_head[:12]}")
            except Exception as e:
                self._log(f"Failed to restore Ascend HEAD: {e}", "warn")

    def _print_progress(self, current: int, total: int, prefix: str = "Progress"):
        """Print progress bar."""
        bar_width = 40
        filled = int((current / total) * bar_width) if total > 0 else 0
        empty = bar_width - filled
        pct = (current / total * 100) if total > 0 else 0
        bar = "█" * filled + "░" * empty
        end = "\r" if current < total else "\n"
        print(f"\r\033[94m[{prefix}]\033[0m [{bar}] {pct:.1f}% ({current}/{total})  ", end=end)

    def build_commit_timeline(self) -> list[CommitGroup]:
        """Build timeline of commit groups from org members and upstream commits.

        Groups are formed by:
          1. PR commits -> 1 group per PR (with PR number and title)
          2. Non-PR consecutive commits by same user -> 1 session group

        Upstream commits (non-org-member) are also included if --upstream-commits
        is specified, or auto-detected from the commit history between org member
        commits.
        """
        self._log("Building commit timeline from cache...")
        return self._build_commit_timeline(ttl_seconds=self._timeline_ttl)

    def run_benchmark_for_group(self, group: CommitGroup) -> Optional[BenchmarkResult]:
        """Run benchmark for a commit group."""
        latest_commit = group.latest_commit
        sha_short = latest_commit[:12]
        run_id = f"group-{group.group_id}-{sha_short}"

        # Skip only if status is success (don't re-run successful benchmarks)
        if group.group_id in self.checkpoint.groups_processed:
            status = self.checkpoint.groups_processed[group.group_id]["status"]
            # In resume mode, we allow re-running failed benchmarks
            if status == "success":
                self._log(f"Skipping {group.group_id} (already succeeded)")
                return None
            # For failed status, only skip if NOT in resume mode
            if status == "failed" and not self.config.resume_mode:
                self._log(f"Skipping {group.group_id} (status: {status})")
                return None
            # In resume mode with failed status, we proceed to re-run

        group_label = f"PR #{group.pr_number}" if group.pr_number else group.group_id
        self._log(f"Running benchmark for {run_id} ({group.group_type}: {group_label})")
        self._log(f"  scenario: {self.config.benchmark_scenario}, model: {self._effective_model} [{self._model_resolution_mode}]")
        self._log(f"  chips: {self.config.chip_count}, submitter: {group.submitter}")
        self._log(f"  commits: {len(group.commits)}, latest: {sha_short}")

        # Prepare output directory
        output_dir = self.config.benchmark_repo / "submissions" / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.dry_run:
            return BenchmarkResult(throughput_tps=0, ttft_ms=0, success=True, output_dir=str(output_dir))

        # Build the benchmark CLI command (conda_python needed by both branches)
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        conda_python = f"{conda_prefix}/bin/python" if conda_prefix else sys.executable
        if not Path(conda_python).is_file():
            conda_python = sys.executable
        cmd = [
            conda_python, "-m", "vllm_hust_benchmark.cli", "run-ascend-ci",
            "--execute",
            "--scenario", self.config.benchmark_scenario,
            "--model", self._effective_model,
            "--chip-count", str(self.config.chip_count),
            "--run-id", run_id,
        ]

        if self._worktree_mgr is None:
            # ---- Legacy in-place git checkout mode ----
            if not self._git_checkout_fallback(latest_commit):
                self._log(f"Failed to checkout {sha_short}, skipping", "error")
                return None

            # Also checkout vllm-ascend-hust to a compatible commit
            ascend_repo = self.config.vllm_ascend_hust_repo
            if ascend_repo and ascend_repo.is_dir():
                ascend_sha = self._resolve_ascend_sha(group)
                fallback_ascend_sha = ""
                commit_date = group.latest_date
                if commit_date:
                    fallback_ascend_sha = self._get_sha_at_time(ascend_repo, commit_date)

                if ascend_sha:
                    if not self._git_checkout_ascend(ascend_repo, ascend_sha):
                        if fallback_ascend_sha and fallback_ascend_sha != ascend_sha:
                            self._log(
                                f"  [Ascend] PR SHA {ascend_sha[:12]} failed, "
                                f"trying time-based fallback {fallback_ascend_sha[:12]}",
                                "warn",
                            )
                            self._git_checkout_ascend(ascend_repo, fallback_ascend_sha)

            env = self._build_legacy_env(output_dir)
            return self._run_benchmark_cli(cmd, conda_python, output_dir, env)

        # ---- Va/Vb worktree isolation mode ----
        ascend_sha = self._resolve_ascend_sha(group)

        # Pre-compute time-based fallback SHA (always on origin/main, always fetchable)
        fallback_ascend_sha = ""
        commit_date = group.latest_date
        if commit_date and self._va_vllm_ascend:
            fallback_ascend_sha = self._get_sha_at_time(self._va_vllm_ascend, commit_date)

        # Quick check: if PR SHA is already local, no fallback needed
        pr_sha_available = (
            self._worktree_mgr._sha_exists_locally(self._va_vllm_ascend, ascend_sha)
            if self._va_vllm_ascend else False
        )

        if not pr_sha_available and fallback_ascend_sha:
            # PR SHA likely not fetchable — use time-based SHA directly
            self._log(f"  [Ascend] PR SHA {ascend_sha[:12]} not local, using time-based SHA {fallback_ascend_sha[:12]} (at {commit_date[:10]})")
            ascend_sha = fallback_ascend_sha
            fallback_ascend_sha = ""  # no need for fallback now

        worktrees: dict[str, Path] = {}

        try:
            self._log(f"Provisioning Vb worktrees (vllm-hust@{latest_commit[:12]}, vllm-ascend-hust@{ascend_sha[:12]})")
            worktrees = self._worktree_mgr.provision(
                vllm_hust_sha=latest_commit,
                vllm_ascend_sha=ascend_sha,
                conda_python=conda_python,
                fallback_ascend_sha=fallback_ascend_sha,
            )
            env = self._build_worktree_env(worktrees, output_dir)

            # Pre-flight: verify ascend plugin can be imported with the
            # vllm-hust version under test.  Catches cross-repo API
            # incompatabilities (e.g. ascend imports a symbol that does
            # not exist in this vllm-hust commit) BEFORE spending minutes
            # running the benchmark.
            preflight_ok = self._preflight_ascend_import(conda_python, env)
            if not preflight_ok:
                # --- Retry with fallback ascend SHA if available ---
                if fallback_ascend_sha and fallback_ascend_sha != ascend_sha:
                    self._log(
                        f"  [Ascend] Preflight failed for {ascend_sha[:12]}, "
                        f"retrying with fallback time-based SHA {fallback_ascend_sha[:12]}",
                        "warn",
                    )
                    self._worktree_mgr.cleanup(worktrees)
                    worktrees = self._worktree_mgr.provision(
                        vllm_hust_sha=latest_commit,
                        vllm_ascend_sha=fallback_ascend_sha,
                        conda_python=conda_python,
                    )
                    env = self._build_worktree_env(worktrees, output_dir)
                    preflight_ok = self._preflight_ascend_import(conda_python, env)

                if not preflight_ok:
                    self._log(
                        f"Skipping {run_id}: vllm-ascend-hust is incompatible "
                        f"with vllm-hust@{latest_commit[:12]}",
                        "error",
                    )
                    return None

            return self._run_benchmark_cli(cmd, conda_python, output_dir, env)


        finally:
            if worktrees:
                self._worktree_mgr.cleanup(worktrees)

    def _resolve_ascend_sha(self, group: CommitGroup) -> str:
        """Resolve the vllm-ascend-hust commit SHA for a group.

        Strategy:
        1. If group has a PR number, look up the PR's merge commit SHA in vllm-ascend-hust
        2. If that PR exists in vllm-ascend-hust, use its merge SHA
        3. Otherwise (no PR or PR not in ascend repo): find the vllm-ascend-hust commit
           at the same point in time as the vllm-hust commit (time-based resolution)
        """
        if group.pr_number is not None:
            # Try to get the corresponding merge SHA from vllm-ascend-hust
            try:
                ascend_sha = self.github.get_pr_merge_sha("vLLM-HUST/vllm-ascend-hust", group.pr_number)
                if ascend_sha:
                    self._log(f"  [Ascend] PR #{group.pr_number} merge SHA in vllm-ascend-hust: {ascend_sha[:12]}")
                    return ascend_sha
            except Exception as e:
                self._log(f"  [Ascend] Failed to get PR #{group.pr_number} from vllm-ascend-hust: {e}", "warn")
            self._log(f"  [Ascend] PR #{group.pr_number} not found in vllm-ascend-hust, falling back to time-based lookup")

        # Time-based fallback: find the vllm-ascend-hust commit at the same point in time
        commit_date = group.latest_date
        if not commit_date:
            self._log(f"  [Ascend] No commit date available, using origin/main HEAD", "warn")
            return self._get_main_head_sha(self._va_vllm_ascend)

        self._log(f"  [Ascend] Time-based lookup: vllm-hust commit date={commit_date[:10]}, finding vllm-ascend-hust commit...")
        sha = self._get_sha_at_time(self._va_vllm_ascend, commit_date)
        if sha:
            self._log(f"  [Ascend] Found vllm-ascend-hust commit at {commit_date[:10]}: {sha[:12]}")
            return sha

        # Last resort: origin/main HEAD
        self._log(f"  [Ascend] Time-based lookup failed, using origin/main HEAD", "warn")
        return self._get_main_head_sha(self._va_vllm_ascend)

    def _get_sha_at_time(self, repo_path: Path | None, iso_date: str) -> str:
        """Find the latest commit on origin/main before a given ISO timestamp.

        Uses `git log --before=<date> -1 origin/main` to find the closest commit.
        """
        if repo_path is None:
            return ""
        try:
            # Fetch first to ensure we have latest remote state
            subprocess.run(["git", "fetch", "origin"], cwd=repo_path, capture_output=True, timeout=30)
            result = subprocess.run(
                ["git", "log", f"--before={iso_date}", "-1", "--format=%H", "origin/main"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    def _get_main_head_sha(self, repo_path: Path | None) -> str:
        """Get the HEAD commit SHA of origin/main for a repo."""
        if repo_path is None:
            return ""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "origin/main"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return ""

    # -------------------------------------------------------------------------
    # Timeline Cache
    # -------------------------------------------------------------------------

    def _load_timeline_cache(self) -> TimelineCache | None:
        """Load commit timeline from cache file if it exists and is not stale."""
        cache_path = Path(self.config.timeline_cache_file)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path) as f:
                data = json.load(f)
            return TimelineCache.from_dict(data)
        except Exception as e:
            self._log(f"Cache load failed (will fetch from GitHub): {e}", "warn")
            return None

    def _save_timeline_cache(self, cache: TimelineCache) -> None:
        """Save commit timeline to cache file."""
        cache_path = Path(self.config.timeline_cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_path, "w") as f:
                json.dump(cache.to_dict(), f, indent=2)
            self._log(f"Timeline cached to {cache_path}")
        except Exception as e:
            self._log(f"Cache save failed (non-fatal): {e}", "warn")

    def _build_commit_timeline(self, ttl_seconds: int = 86400) -> list[CommitGroup]:
        """Build the commit timeline, using cache when available.

        Cache strategy:
        - Check cache first; if fresh and not forced-refresh, return cached data
        - Otherwise fetch from GitHub, cache result, and return
        - Force-refresh triggered by --refresh-timeline flag

        Args:
            ttl_seconds: Cache TTL in seconds (default 24h).
        """
        # Check cache
        if not self.config.refresh_timeline:
            cached = self._load_timeline_cache()
            if cached and not cached.is_stale(ttl_seconds):
                self._log(f"Timeline cache hit (cached {cached.cached_at}), using cached data")
                return self._reconstruct_groups_from_cache(cached)

        self._log("Building commit timeline from GitHub...")
        groups = self._fetch_timeline_from_github()

        # Save cache on successful fetch
        cache = self._create_cache_from_groups(groups)
        self._save_timeline_cache(cache)

        return groups

    def _reconstruct_groups_from_cache(self, cache: TimelineCache) -> list[CommitGroup]:
        """Reconstruct CommitGroup list from cached data."""
        # Rebuild pr_cache lookup: sha -> pr_info
        pr_cache_map: dict[str, dict] = {}
        for entry in cache.pr_cache:
            pr_cache_map[entry["sha"]] = entry

        # Rebuild all_commits list
        all_commits: list[tuple[str, str, str, bool]] = []
        for entry in cache.all_commits:
            all_commits.append((entry["date"], entry["sha"], entry["user"], entry["is_org_member"]))

        # Group commits by PR
        hard_excluded = set(m.strip() for m in self.config.excluded_members.split(","))
        target_members = set(cache.org_members) - hard_excluded

        pr_groups: dict[int, list[CommitInfo]] = {}
        non_pr_commits: list[tuple[str, str, str, bool]] = []

        for date, sha, user, is_org in all_commits:
            pr_info = pr_cache_map.get(sha)
            if pr_info:
                pr_num = pr_info["number"]
                if pr_num not in pr_groups:
                    pr_groups[pr_num] = []
                pr_groups[pr_num].append(CommitInfo(
                    sha=sha, date=date, user=user,
                    pr_number=pr_num,
                    message=""
                ))
            else:
                non_pr_commits.append((date, sha, user, is_org))

        groups: list[CommitGroup] = []
        for pr_num, commits in pr_groups.items():
            commits.sort(key=lambda c: c.date)
            is_org = any(c.user in target_members for c in commits)
            pr_info_sample = pr_cache_map.get(commits[0].sha, {})
            groups.append(CommitGroup(
                group_id=f"pr-{pr_num}",
                group_type="pr",
                commits=commits,
                is_org_member=is_org,
                pr_number=pr_num,
                pr_title=pr_info_sample.get("title", "") if pr_info_sample else "",
            ))

        current_group: list[CommitInfo] = []
        current_user = ""
        current_is_org = False
        for date, sha, user, is_org in non_pr_commits:
            if user != current_user or not current_group:
                if current_group:
                    groups.append(CommitGroup(
                        group_id=f"session-{current_group[0].user}-{current_group[0].sha[:12]}",
                        group_type="session",
                        commits=current_group,
                        is_org_member=current_is_org,
                    ))
                current_group = []
                current_user = user
                current_is_org = is_org
            current_group.append(CommitInfo(sha=sha, date=date, user=user))

        if current_group:
            groups.append(CommitGroup(
                group_id=f"session-{current_group[0].user}-{current_group[0].sha[:12]}",
                group_type="session",
                commits=current_group,
                is_org_member=current_is_org,
            ))

        groups.sort(key=lambda g: g.latest_date)
        return groups

    def _create_cache_from_groups(self, groups: list[CommitGroup]) -> TimelineCache:
        """Serialize CommitGroup list into a TimelineCache."""
        import datetime
        all_commits: list[dict] = []
        pr_cache_entries: list[dict] = []
        org_members: set[str] = set()
        pr_cache_map: dict[str, dict] = {}  # sha -> pr_info

        for group in groups:
            for c in group.commits:
                all_commits.append({
                    "date": c.date,
                    "sha": c.sha,
                    "user": c.user,
                    "is_org_member": group.is_org_member,
                })
                org_members.add(c.user)

            if group.pr_number is not None:
                # Collect all SHAs in this PR group for PR association
                for c in group.commits:
                    pr_cache_map[c.sha] = {
                        "sha": c.sha,
                        "number": group.pr_number,
                        "title": group.pr_title,
                        "user": c.user,
                    }

        return TimelineCache(
            org_members=sorted(org_members),
            all_commits=all_commits,
            pr_cache=list(pr_cache_map.values()),
            cached_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    def _fetch_timeline_from_github(self) -> list[CommitGroup]:
        """Fetch commit timeline from GitHub API (no cache).
        
        This is the underlying fetch logic called by _build_commit_timeline
        when cache is missing, stale, or force-refresh is requested.
        """
        # Parse excluded members (hard-coded exclusions)
        hard_excluded = set(m.strip() for m in self.config.excluded_members.split(","))

        # Get org members from GitHub
        org_members: set[str] = set()
        try:
            org_members = set(self.github.get_org_members())
            self._log(f"Found {len(org_members)} org members")
        except Exception as e:
            self._log(f"Failed to fetch org members: {e}", "warn")

        # Target members = org members minus hard exclusions
        target_members = org_members - hard_excluded
        self._log(f"Target members (after exclusions): {target_members}")

        # Collect all commits: (date, sha, user, is_org_member)
        all_commits: list[tuple[str, str, str, bool]] = []

        # Fetch org member commits
        for member in target_members:
            self._log(f"Fetching commits for {member}...")
            try:
                commits = self.github.get_user_commits(
                    "vLLM-HUST/vllm-hust", member, self.config.since_date
                )
                for c in commits:
                    all_commits.append((c.date, c.sha, c.user, True))
                self._log(f"  Found {len(commits)} commits for {member}")
            except Exception as e:
                self._log(f"Failed to fetch commits for {member}: {e}", "warn")

        # Fetch upstream commits if specified
        upstream_shas = set()
        if self.config.upstream_commits:
            upstream_shas = set(s.strip() for s in self.config.upstream_commits.split(",") if s.strip())
            self._log(f"Upstream commits specified: {len(upstream_shas)} SHAs")
            for sha in upstream_shas:
                try:
                    data = self.github._request(f"/repos/vLLM-HUST/vllm-hust/commits/{sha}")
                    date = data["commit"]["author"]["date"]
                    user = data.get("author", {}).get("login", data["commit"]["author"]["name"])
                    all_commits.append((date, sha, user, False))
                    self._log(f"  Added upstream commit {sha[:12]} by {user}")
                except Exception as e:
                    self._log(f"Failed to fetch upstream commit {sha[:12]}: {e}", "warn")

        if not all_commits:
            self._log("No commits found!", "warn")
            return []

        # Sort by date
        all_commits.sort(key=lambda x: x[0])
        self._log(f"Total commits collected: {len(all_commits)}")

        # Detect PRs for all commits
        self._log("Detecting PR associations...")
        pr_cache: dict[str, Optional[dict]] = {}
        for date, sha, user, is_org in all_commits:
            pr_info = self.github.get_commit_pr("vLLM-HUST/vllm-hust", sha)
            if pr_info:
                pr_cache[sha] = pr_info
                self._log(f"  {sha[:12]} -> PR #{pr_info['number']}: {pr_info['title'][:50]}")

        # Group commits
        groups: list[CommitGroup] = []

        # First, group PR commits together
        pr_groups: dict[int, list[CommitInfo]] = {}
        non_pr_commits: list[tuple[str, str, str, bool]] = []

        for date, sha, user, is_org in all_commits:
            pr_info = pr_cache.get(sha)
            if pr_info:
                pr_num = pr_info["number"]
                if pr_num not in pr_groups:
                    pr_groups[pr_num] = []
                pr_groups[pr_num].append(CommitInfo(
                    sha=sha, date=date, user=user,
                    pr_number=pr_num,
                    message=""
                ))
            else:
                non_pr_commits.append((date, sha, user, is_org))

        # Create PR groups
        for pr_num, commits in pr_groups.items():
            commits.sort(key=lambda c: c.date)
            is_org = any(c.user in target_members for c in commits)
            pr_info_sample = pr_cache.get(commits[0].sha, {})
            groups.append(CommitGroup(
                group_id=f"pr-{pr_num}",
                group_type="pr",
                commits=commits,
                is_org_member=is_org,
                pr_number=pr_num,
                pr_title=pr_info_sample.get("title", "") if pr_info_sample else "",
            ))
            self._log(f"  PR group #{pr_num}: {len(commits)} commits, org={is_org}")

        # Group non-PR commits by session (consecutive same user)
        current_group: list[CommitInfo] = []
        current_user = ""
        current_is_org = False

        for date, sha, user, is_org in non_pr_commits:
            if user != current_user or not current_group:
                if current_group:
                    groups.append(CommitGroup(
                        group_id=f"session-{current_group[0].user}-{current_group[0].sha[:12]}",
                        group_type="session",
                        commits=current_group,
                        is_org_member=current_is_org,
                    ))
                current_group = []
                current_user = user
                current_is_org = is_org
            current_group.append(CommitInfo(sha=sha, date=date, user=user))

        if current_group:
            groups.append(CommitGroup(
                group_id=f"session-{current_group[0].user}-{current_group[0].sha[:12]}",
                group_type="session",
                commits=current_group,
                is_org_member=current_is_org,
            ))

        # Sort all groups by latest date
        groups.sort(key=lambda g: g.latest_date)
        return groups

    def _build_worktree_env(
        self,
        worktrees: dict[str, Path],
        output_dir: Path,
    ) -> dict[str, str]:
        """Build environment for Vb (worktree-based) benchmark run.

        Va paths provide stable infrastructure (CI scripts, constraints files).
        Vb paths provide code under test (editable installs).
        """
        env = os.environ.copy()

        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        conda_bin = f"{conda_prefix}/bin" if conda_prefix else ""
        conda_python = f"{conda_bin}/python" if conda_bin else sys.executable

        if conda_bin:
            existing_path = env.get("PATH", "")
            env["PATH"] = conda_bin + (":" + existing_path if existing_path else "")
            env["PYTHON_BIN"] = conda_python

        # Va paths: infrastructure lives in the stable Va worktree
        va_vllm_hust = self._va_vllm_hust or self.config.vllm_hust_repo
        va_vllm_ascend = self._va_vllm_ascend

        # Vb paths: code under test
        vb_vllm_hust = worktrees["vllm_hust"]
        vb_vllm_ascend = worktrees["vllm_ascend"]

        env["VLLM_HUST_REPO"] = str(vb_vllm_hust)
        env["VLLM_ASCEND_HUST_REPO"] = str(vb_vllm_ascend)
        env["VLLM_HUST_WORKSPACE_ROOT"] = str(self.config.benchmark_repo.parent)
        # Explicitly set VLLM_HUST_WEBSITE_REPO so the CI script uses the
        # correct sibling path instead of deriving it from WORKSPACE_ROOT.
        # When benchmark_repo is the scripts/ subdirectory, its parent is the
        # actual benchmark repo, and the workspace root is the grand-parent.
        website_candidate = self.config.benchmark_repo.parent / "vllm-hust-website"
        if website_candidate.is_dir():
            env["VLLM_HUST_WEBSITE_REPO"] = str(website_candidate)
        else:
            # Fallback: use benchmark_repo itself as base (for non-standard layouts)
            env["VLLM_HUST_WEBSITE_REPO"] = str(self.config.benchmark_repo / "vllm-hust-website")

        # CI script: use Va's infrastructure (always stable)
        if va_vllm_hust:
            ci_script = va_vllm_hust / ".github" / "workflows" / "scripts" / "run_ascend_benchmark_ci.sh"
            if ci_script.is_file():
                env["VLLM_HUST_CI_SCRIPT"] = str(ci_script)

        # Constraints file: use Va's data dir (always stable)
        if va_vllm_ascend:
            constraints = va_vllm_ascend / ".github" / "workflows" / "data" / "random-online-ci-constraints.json"
            if constraints.is_file():
                env["BENCH_CONSTRAINTS_FILE"] = str(constraints)

        # Direct outputs to benchmark repo's submissions dir
        env["SUBMISSIONS_ROOT"] = str(output_dir.parent)
        env["RESULT_ROOT"] = str(output_dir)
        env["SAME_SPEC_BENCHMARK_ENABLED"] = "0"
        env["_VLLM_INTERNAL_ENV_REEXEC_DONE"] = "1"

        # PYTHONPATH: put Vb worktree packages FIRST so they override conda env
        python_path = str(vb_vllm_hust) + ":" + str(vb_vllm_ascend)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = python_path + (":" + existing_pp if existing_pp else "")

        # CANN env
        if self._cann_env:
            for key in ("LD_LIBRARY_PATH", "PATH"):
                cann_val = self._cann_env.get(key, "")
                if cann_val:
                    existing = env.get(key, "")
                    env[key] = cann_val + (":" + existing if existing else "")
            for key, val in self._cann_env.items():
                if key.startswith(("ASCEND_", "ATB_")):
                    env[key] = val
            ld_preload = env.get("LD_PRELOAD", "")
            if ld_preload:
                env["LD_PRELOAD"] = ":".join(p for p in ld_preload.split(":") if p)

        return env

    def _build_legacy_env(self, output_dir: Path) -> dict[str, str]:
        """Build environment for legacy in-place checkout mode."""
        env = os.environ.copy()

        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            conda_bin = f"{conda_prefix}/bin"
            existing_path = env.get("PATH", "")
            if conda_bin not in existing_path:
                env["PATH"] = conda_bin + (":" + existing_path if existing_path else "")
            env["PYTHON_BIN"] = f"{conda_bin}/python"

        if self.config.vllm_hust_repo:
            env["VLLM_HUST_REPO"] = str(self.config.vllm_hust_repo)
            env["VLLM_HUST_WORKSPACE_ROOT"] = str(self.config.vllm_hust_repo.parent)

        if getattr(self, "_ci_script_stable_path", None) and self._ci_script_stable_path.is_file():
            env["VLLM_HUST_CI_SCRIPT"] = str(self._ci_script_stable_path)
        if getattr(self, "_constraints_file_stable_path", None) and self._constraints_file_stable_path.is_file():
            env["BENCH_CONSTRAINTS_FILE"] = str(self._constraints_file_stable_path)

        env["SUBMISSIONS_ROOT"] = str(output_dir.parent)
        env["RESULT_ROOT"] = str(output_dir)
        env["SAME_SPEC_BENCHMARK_ENABLED"] = "0"
        env["_VLLM_INTERNAL_ENV_REEXEC_DONE"] = "1"

        if self._cann_env:
            for key in ("LD_LIBRARY_PATH", "PATH"):
                cann_val = self._cann_env.get(key, "")
                if cann_val:
                    existing = env.get(key, "")
                    env[key] = cann_val + (":" + existing if existing else "")
            for key, val in self._cann_env.items():
                if key.startswith(("ASCEND_", "ATB_")):
                    env[key] = val
            ld_preload = env.get("LD_PRELOAD", "")
            if ld_preload:
                env["LD_PRELOAD"] = ":".join(p for p in ld_preload.split(":") if p)

        python_path = str(self.config.benchmark_repo)
        existing_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = python_path + (":" + existing_pp if existing_pp else "")

        return env

    def _preflight_ascend_import(
        self,
        conda_python: str,
        env: dict[str, str],
    ) -> bool:
        """Check that vllm_ascend can be imported in the current env.

        Catches cross-repo API incompatibilities (e.g. the ascend plugin
        imports ``vllm_is_batch_invariant`` from vllm but the vllm-hust
        commit under test does not expose that symbol) *before* the full
        benchmark is launched.

        Performs two checks:
        1. Top-level ``import vllm_ascend`` (catches basic import failures)
        2. Deep-import scan of ``ascend_config.py`` — detects lazy imports
           inside ``AscendConfig.__init__`` that would only trigger at engine
           startup time (e.g. ``from vllm…import vllm_is_batch_invariant``)

        Returns True if all checks pass (or cannot be verified),
        False if they definitively fail.
        """
        try:
            result = subprocess.run(
                [conda_python, "-c", "import vllm_ascend"],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            if result.returncode != 0:
                # Import failed — extract the missing symbol for diagnostics
                stderr = result.stderr or ""
                self._log(f"  [preflight] vllm_ascend import FAILED:", "error")
                for line in stderr.strip().splitlines()[-5:]:
                    self._log(f"    {line}", "error")
                return False
        except subprocess.TimeoutExpired:
            self._log("  [preflight] import check timed out, proceeding anyway", "warn")
            return True  # assume OK on timeout
        except Exception as exc:
            self._log(f"  [preflight] import check error: {exc}, proceeding anyway", "warn")
            return True  # assume OK on unexpected error

        # --- Deep-import scan ---
        # AscendConfig.__init__ performs lazy imports from vllm that are NOT
        # triggered by a simple ``import vllm_ascend``.  Scan ascend_config.py
        # for ``from vllm…import …`` statements and verify each symbol exists
        # in the current vllm installation.
        try:
            deep_check_script = textwrap.dedent("""\
                import importlib, ast, sys, os

                ascend_path = os.environ.get("VLLM_ASCEND_HUST_REPO", "")
                if not ascend_path:
                    sys.exit(0)  # cannot determine path, skip check

                config_file = os.path.join(ascend_path, "vllm_ascend", "ascend_config.py")
                if not os.path.isfile(config_file):
                    sys.exit(0)

                with open(config_file) as f:
                    tree = ast.parse(f.read())

                missing = []
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    if not (node.module or "").startswith("vllm."):
                        continue
                    for alias in node.names:
                        try:
                            mod = importlib.import_module(node.module)
                            if not hasattr(mod, alias.name):
                                missing.append(f"{node.module}.{alias.name}")
                        except Exception:
                            pass

                if missing:
                    for m in missing:
                        print(f"DEEP_IMPORT_MISSING:{m}", file=sys.stderr)
                    sys.exit(1)
            """)
            deep_result = subprocess.run(
                [conda_python, "-c", deep_check_script],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            if deep_result.returncode != 0:
                stderr = deep_result.stderr or ""
                missing_symbols = [
                    line.split(":", 1)[1]
                    for line in stderr.strip().splitlines()
                    if line.startswith("DEEP_IMPORT_MISSING:")
                ]
                if missing_symbols:
                    self._log(
                        f"  [preflight] Deep-import check FAILED: "
                        f"missing symbol(s) in vllm: {', '.join(missing_symbols)}",
                        "error",
                    )
                    self._log(
                        "  [preflight] The ascend plugin imports symbols that do "
                        "not exist in this vllm-hust version.",
                        "error",
                    )
                else:
                    self._log("  [preflight] Deep-import check FAILED", "error")
                    for line in stderr.strip().splitlines()[-5:]:
                        self._log(f"    {line}", "error")
                return False
            self._log("  [preflight] vllm_ascend import OK (deep-import scan passed)", "success")
        except Exception as exc:
            # Deep check is best-effort — don't block on its failure
            self._log(f"  [preflight] vllm_ascend import OK (deep-import scan skipped: {exc})", "success")

        return True

    def _run_benchmark_cli(
        self,
        cmd: list[str],
        conda_python: str,
        output_dir: Path,
        env: dict[str, str],
    ) -> Optional[BenchmarkResult]:
        """Run the benchmark CLI with the given environment."""
        self._log(f"  cmd: {shlex.join(cmd)}")
        self._log(f"  cwd: {self.config.benchmark_repo}")
        self._log("  benchmark started, streaming output...")
        sep = "-" * 60
        self._tee_print(sep)

        start_time = time.monotonic()
        output_lines: list[str] = []
        max_tail_lines = 50

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.config.benchmark_repo,
                env=env,
                text=True,
                bufsize=1,
            )

            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                output_lines.append(line)
                if len(output_lines) > max_tail_lines:
                    output_lines.pop(0)
                self._tee_print(line)

            proc.wait()
            elapsed = time.monotonic() - start_time

        except OSError as e:
            self._log(f"Failed to launch benchmark process: {e}", "error")
            return None

        self._tee_print(sep)

        if proc.returncode != 0:
            elapsed = time.monotonic() - start_time
            self._log(f"Benchmark failed (exit code {proc.returncode}, elapsed {elapsed:.1f}s)", "error")
            if output_lines:
                self._log("--- last output from benchmark ---", "warn")
                for tail_line in output_lines[-20:]:
                    self._tee_print(tail_line)
                self._log("--- end of output ---", "warn")

            # Detect ascend plugin version incompatibility
            full_output = "\n".join(output_lines)
            if "ImportError" in full_output and "cannot import name" in full_output:
                self._log(
                    "Root cause: vllm-ascend-hust worktree is incompatible with "
                    "the vllm-hust commit under test. The ascend plugin imports a "
                    "symbol that does not exist in this vllm-hust version.",
                    "error",
                )
                self._log(
                    "Fix: use a vllm-ascend-hust commit that is compatible with "
                    "the vllm-hust commit, or update the time-based SHA resolution "
                    "to pick an older ascend commit.",
                    "warn",
                )
            return None

        elapsed = time.monotonic() - start_time
        self._log(f"Benchmark completed in {elapsed:.1f}s")

        leaderboard_file = output_dir / "run_leaderboard.json"
        if leaderboard_file.exists():
            with open(leaderboard_file) as f:
                data = json.load(f)
            return BenchmarkResult(
                throughput_tps=data.get("metrics", {}).get("throughput_tps", 0),
                ttft_ms=data.get("metrics", {}).get("ttft_ms", 0),
                success=True,
                output_dir=str(output_dir),
            )

        self._log(f"Leaderboard file not found: {leaderboard_file}", "warn")
        return None

    def calculate_attribution(self, group: CommitGroup, result: BenchmarkResult) -> Optional[AttributionEntry]:
        """Calculate attribution for an org member contribution."""
        # Find last upstream baseline
        upstream_commits = sorted(
            self.checkpoint.upstream_baselines.keys(),
            key=lambda x: self.checkpoint.upstream_baselines[x].get("timestamp", "")
        )

        if not upstream_commits:
            self._log(f"No upstream baseline for {group.group_id}", "warn")
            return None

        last_upstream = upstream_commits[-1]
        baseline = self.checkpoint.upstream_baselines[last_upstream]

        upstream_tps = baseline.get("throughput_tps", 0)
        upstream_ttft = baseline.get("ttft_ms", 0)

        if upstream_tps == 0:
            return None

        org_tps = result.throughput_tps
        org_ttft = result.ttft_ms

        delta_tps = org_tps - upstream_tps
        improvement_pct = round((delta_tps / upstream_tps) * 100, 2)
        ttft_improvement_pct = round(((upstream_ttft - org_ttft) / upstream_ttft) * 100, 2) if upstream_ttft else 0

        return AttributionEntry(
            org_member=group.submitter,
            group_id=group.group_id,
            group_type=group.group_type,
            pr_number=None,
            commit_count=len(group.commits),
            latest_commit=group.latest_commit,
            latest_commit_date=group.latest_date,
            upstream_baseline_commit=last_upstream,
            upstream_throughput_tps=upstream_tps,
            org_throughput_tps=org_tps,
            delta_tps=round(delta_tps, 2),
            improvement_percent=improvement_pct,
            ttft_improvement_percent=ttft_improvement_pct,
            attribution_timestamp=datetime.utcnow().isoformat() + "Z"
        )

    def commit_and_push(self, message: str) -> bool:
        """Commit and push changes to GitHub.

        Returns True only if both commit AND push succeed.
        """
        if self.config.dry_run:
            self._log(f"[DRY RUN] Would commit: {message}")
            return True

        try:
            # Only add files that exist to avoid git add exit 128
            add_targets = ["submissions/"]
            attr_file = Path(self.config.attribution_file)
            if attr_file.is_file():
                add_targets.append(str(attr_file))
            subprocess.run(["git", "add"] + add_targets,
                         cwd=self.config.benchmark_repo, capture_output=True, check=True)

            result = subprocess.run(["git", "commit", "-m", message],
                                   cwd=self.config.benchmark_repo, capture_output=True, check=True)

            # Detect the current branch — commits are on whatever branch is
            # checked out, which may differ from config.benchmark_branch
            # (e.g. workspace branches like ws/benchmark_backfill).
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.config.benchmark_repo, capture_output=True, text=True,
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else self.config.benchmark_branch
            if current_branch == "HEAD":
                # Detached HEAD — push to configured branch instead
                current_branch = self.config.benchmark_branch

            # Try push via SSH first
            push_result = subprocess.run(
                ["git", "push", "origin", current_branch],
                cwd=self.config.benchmark_repo, capture_output=True, text=True,
            )

            if push_result.returncode == 0:
                self._log(f"  [push] Pushed {current_branch} to origin successfully", "success")
                return True

            # SSH push failed — try token-based push
            self._log(
                f"  [push] SSH push failed (exit {push_result.returncode}), "
                f"trying token-based push...",
                "warn",
            )
            token = getattr(self.github, "token", "") or ""
            if token:
                original_url_result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.config.benchmark_repo, capture_output=True, text=True,
                )
                original_url = original_url_result.stdout.strip()

                remote = f"https://x-access-token:{token}@github.com/vLLM-HUST/vllm-hust-benchmark.git"
                subprocess.run(
                    ["git", "remote", "set-url", "origin", remote],
                    cwd=self.config.benchmark_repo, capture_output=True,
                )
                token_result = subprocess.run(
                    ["git", "push", "origin", current_branch],
                    cwd=self.config.benchmark_repo, capture_output=True, text=True,
                )
                # Restore original URL
                subprocess.run(
                    ["git", "remote", "set-url", "origin", original_url],
                    cwd=self.config.benchmark_repo, capture_output=True,
                )
                if token_result.returncode == 0:
                    self._log(f"  [push] Token-based push succeeded", "success")
                    return True
                self._log(
                    f"  [push] Token-based push also failed "
                    f"(exit {token_result.returncode}): "
                    f"{token_result.stderr.strip()[:200]}",
                    "error",
                )
            else:
                self._log(
                    "  [push] No GitHub token available for fallback push",
                    "error",
                )

            self._log(
                f"  [push] WARNING: commit created but NOT pushed to remote. "
                f"Run 'git push origin {current_branch}' manually "
                f"in {self.config.benchmark_repo} to publish results.",
                "error",
            )
            return False
        except subprocess.CalledProcessError as e:
            self._log(f"Git operation failed: {e}", "error")
            return False
        except Exception as e:
            self._log(f"Push failed: {e}", "error")
            return False

    def _publish_to_website_snapshots(self) -> bool:
        """Aggregate submissions into leaderboard-data/snapshots/ and commit/push.

        Required so the website's sync-leaderboard-data workflow has fresh
        snapshots to copy into website/data/. Without this step, per-submission
        artifacts in submissions/ are never aggregated into the canonical
        snapshots consumed by the website.
        """
        if self.config.dry_run:
            self._log(
                "[DRY RUN] Would aggregate submissions/ into leaderboard-data/snapshots/",
                "warn",
            )
            return True

        submissions_dir = self.config.benchmark_repo / "submissions"
        snapshots_dir = self.config.benchmark_repo / "leaderboard-data" / "snapshots"

        if not submissions_dir.is_dir():
            self._log(
                f"No submissions directory at {submissions_dir}; skipping publish.",
                "warn",
            )
            return True

        # Skip when no submission manifests exist (e.g. all groups failed).
        manifest_files = list(submissions_dir.glob("*/leaderboard_manifest.json"))
        if not manifest_files:
            self._log(
                "No submission manifests found; skipping publish-website.",
                "warn",
            )
            return True

        snapshots_dir.mkdir(parents=True, exist_ok=True)

        # Set up environment so the CLI can locate the website repo.
        env = os.environ.copy()
        env["VLLM_HUST_WORKSPACE_ROOT"] = str(self.config.benchmark_repo.parent)
        if getattr(self.config, "vllm_hust_repo", None):
            env["VLLM_HUST_REPO"] = str(self.config.vllm_hust_repo)

        cmd = [
            sys.executable,
            "-m",
            "vllm_hust_benchmark.cli",
            "publish-website",
            "--source-dir",
            str(submissions_dir),
            "--output-dir",
            str(snapshots_dir),
            "--execute",
        ]

        self._log("Publishing website snapshots from submissions/...")
        result = subprocess.run(
            cmd,
            cwd=self.config.benchmark_repo,
            env=env,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self._log(
                f"publish-website failed (exit {result.returncode})",
                "error",
            )
            if result.stderr:
                self._log(result.stderr.strip()[-1000:], "error")
            if result.stdout:
                self._log(result.stdout.strip()[-1000:], "error")
            return False

        # Surface the aggregation summary lines for visibility.
        if result.stdout:
            for line in result.stdout.strip().splitlines()[-6:]:
                self._log(f"  {line}")

        # Commit and push the snapshots separately so the per-group commit
        # history is preserved and snapshots always land on their own commit.
        snapshot_files = [
            "leaderboard-data/snapshots/leaderboard_single.json",
            "leaderboard-data/snapshots/leaderboard_multi.json",
            "leaderboard-data/snapshots/leaderboard_compare.json",
            "leaderboard-data/snapshots/last_updated.json",
        ]
        existing = [p for p in snapshot_files if (self.config.benchmark_repo / p).is_file()]
        if not existing:
            self._log(
                "No snapshot files were produced; skipping commit.",
                "warn",
            )
            return True

        try:
            subprocess.run(
                ["git", "add"] + existing,
                cwd=self.config.benchmark_repo,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "chore: refresh leaderboard snapshots"],
                cwd=self.config.benchmark_repo,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            # Nothing to commit (snapshots unchanged) is not a failure.
            stderr = (e.stderr or b"").decode(errors="ignore") if hasattr(e, "stderr") else ""
            if "nothing to commit" in stderr or "no changes added" in stderr:
                self._log("Snapshots unchanged; no commit needed.", "warn")
                return True
            self._log(f"Failed to commit snapshots: {e}", "error")
            return False

        # Detect branch and push (mirrors commit_and_push push logic).
        branch_result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=self.config.benchmark_repo,
            capture_output=True,
            text=True,
        )
        current_branch = (
            branch_result.stdout.strip()
            if branch_result.returncode == 0
            else self.config.benchmark_branch
        )
        if current_branch == "HEAD":
            current_branch = self.config.benchmark_branch

        push_result = subprocess.run(
            ["git", "push", "origin", current_branch],
            cwd=self.config.benchmark_repo,
            capture_output=True,
            text=True,
        )
        if push_result.returncode == 0:
            self._log(
                f"  [push] Snapshots pushed to {current_branch}",
                "success",
            )
            return True

        self._log(
            f"  [push] Snapshot push failed (exit {push_result.returncode}): "
            f"{push_result.stderr.strip()[:200]}",
            "error",
        )
        self._log(
            f"  [push] Run 'cd {self.config.benchmark_repo} && "
            f"git push origin {current_branch}' manually to publish snapshots.",
            "warn",
        )
        return False

    def _sync_to_hf_write_side(self) -> bool:
        """Upload local snapshots to HF write side using --skip-aggregation.

        This preserves the local aggregation results instead of letting the
        HF workflow re-aggregate from submissions (which can produce different
        results due to submission set differences).
        """
        if self.config.dry_run:
            self._log(
                "[DRY RUN] Would sync leaderboard-data/snapshots/ to HF write side",
                "warn",
            )
            return True

        snapshots_dir = self.config.benchmark_repo / "leaderboard-data" / "snapshots"
        if not snapshots_dir.is_dir():
            self._log(
                f"No snapshots directory at {snapshots_dir}; skipping HF sync.",
                "warn",
            )
            return True

        # Check if snapshot files exist
        snapshot_files = [
            "leaderboard_single.json",
            "leaderboard_multi.json",
            "leaderboard_compare.json",
            "last_updated.json",
        ]
        existing = [f for f in snapshot_files if (snapshots_dir / f).is_file()]
        if not existing:
            self._log("No snapshot files found; skipping HF sync.", "warn")
            return True

        cmd = [
            sys.executable,
            "-m",
            "vllm_hust_benchmark.cli",
            "sync-submission-to-hf",
            "--aggregate-output-dir",
            str(snapshots_dir),
            "--repo-id",
            "intellistream/vllm-hust-benchmark-results",
            "--skip-aggregation",
            "--execute",
        ]

        self._log("Syncing snapshots to HF write side (skip-aggregation mode)...")
        result = subprocess.run(
            cmd,
            cwd=self.config.benchmark_repo,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            self._log(
                f"HF sync failed (exit {result.returncode})",
                "error",
            )
            if result.stderr:
                self._log(result.stderr.strip()[-1000:], "error")
            if result.stdout:
                self._log(result.stdout.strip()[-1000:], "error")
            return False

        # Surface the upload summary lines for visibility.
        if result.stdout:
            for line in result.stdout.strip().splitlines()[-6:]:
                self._log(f"  {line}")

        self._log("  [HF sync] Snapshots uploaded to intellistream/vllm-hust-benchmark-results", "success")
        return True

    def run(self) -> int:
        """Main run loop."""
        self._print_section("vLLM-HUST Member Benchmark (Delta Attribution)")
        self._log(f"Scenario: {self.config.benchmark_scenario}, Model: {self.config.model_name}")
        self._log(f"Since: {self.config.since_date}, Excluded: {self.config.excluded_members}")
        if self.config.dry_run:
            self._log("Mode: DRY RUN", "warn")

        # Clean up stale worktrees from previous crashed runs
        if self._worktree_mgr:
            self._print_section("Checking for Stale Worktrees")
            self._worktree_mgr.cleanup_stale_worktrees()

        # Initialize checkpoint
        if self.config.resume_mode:
            self._log(f"Resuming from: {self.config.checkpoint_file}")
        else:
            self.checkpoint = Checkpoint()

        # Build timeline
        self._print_section("Building Commit Timeline")
        groups = self.build_commit_timeline()
        self._log(f"Found {len(groups)} groups to process")

        # Filter unprocessed or failed (if force flag is set)
        unprocessed = []
        for g in groups:
            if g.group_id not in self.checkpoint.groups_processed:
                unprocessed.append(g)
            elif self.config.resume_mode and self.checkpoint.groups_processed[g.group_id].get("status") == "failed":
                self._log(f"Re-queuing failed group: {g.group_id}", "warn")
                unprocessed.append(g)
        self._log(f"Groups to process: {len(unprocessed)}")
        
        # Always save checkpoint after building timeline (even if no new groups)
        if unprocessed:
            self.checkpoint.to_file(Path(self.config.checkpoint_file))
        else:
            self._log("All groups already processed!")
            # Generate report even when nothing new to process
            self._log("Generating report from existing checkpoint...")
            self._generate_report_from_checkpoint()
            return 0

        # Va/Vb mode: Va stays stable, no git operations on Va
        # Legacy mode (fallback): save HEAD + CI script, then in-place checkout
        if self._worktree_mgr is None:
            self._log("Worktree mode disabled — using legacy in-place git checkout", "warn")
            self._git_save_head()
            self._save_ci_script_copy()

        # Process groups
        self._print_section("Processing Groups")
        success = 0
        failed = 0

        try:
            for i, group in enumerate(unprocessed, 1):
                print()
                self._log(f"=== Group {i}/{len(unprocessed)}: {group.group_id} ===")
                self._print_progress(i, len(unprocessed))

                # For upstream groups, try baseline lookup first
                if not group.is_org_member and self.config.include_upstream:
                    lookup_result = self._try_upstream_lookup(group)
                    if lookup_result:
                        self._log(f"Found existing upstream baseline for {group.group_id} (source: {lookup_result.get('source', 'unknown')})")
                        self.checkpoint.upstream_baselines[group.latest_commit] = {
                            "throughput_tps": lookup_result["throughput_tps"],
                            "ttft_ms": lookup_result["ttft_ms"],
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "source": lookup_result.get("source", "lookup"),
                        }
                        self.checkpoint.groups_processed[group.group_id] = {
                            "status": "success",
                            "processed_at": datetime.utcnow().isoformat() + "Z",
                            "source": "existing_baseline",
                        }
                        self.checkpoint.benchmarks_completed["success"] += 1
                        self.checkpoint.last_updated = datetime.utcnow().isoformat() + "Z"
                        self.checkpoint.to_file(Path(self.config.checkpoint_file))
                        success += 1
                        continue

                # Run benchmark
                result = self.run_benchmark_for_group(group)

                if result and result.success:
                    success += 1
                    self.checkpoint.groups_processed[group.group_id] = {
                        "status": "success",
                        "processed_at": datetime.utcnow().isoformat() + "Z"
                    }
                    self.checkpoint.benchmarks_completed["success"] += 1

                    # Calculate attribution or save upstream baseline
                    if group.is_org_member:
                        entry = self.calculate_attribution(group, result)
                        if entry:
                            # Set PR info if available
                            if group.pr_number:
                                entry.pr_number = group.pr_number
                            self.checkpoint.attribution.append(asdict(entry))
                    else:
                        # Save as upstream baseline
                        self.checkpoint.upstream_baselines[group.latest_commit] = {
                            "throughput_tps": result.throughput_tps,
                            "ttft_ms": result.ttft_ms,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "source": "benchmark_run",
                        }

                    # Save checkpoint and push
                    self.checkpoint.last_updated = datetime.utcnow().isoformat() + "Z"
                    self.checkpoint.to_file(Path(self.config.checkpoint_file))
                    self.commit_and_push(f"chore: add {group.group_id} benchmark")
                else:
                    failed += 1
                    self.checkpoint.groups_processed[group.group_id] = {
                        "status": "failed",
                        "processed_at": datetime.utcnow().isoformat() + "Z"
                    }
                    self.checkpoint.benchmarks_completed["failed"] += 1
                    self.checkpoint.to_file(Path(self.config.checkpoint_file))

                    if self.config.fail_fast:
                        self._log(
                            f"Fail-fast: aborting after first group failure "
                            f"({group.group_id}). Remaining {len(unprocessed) - i} "
                            f"group(s) skipped.",
                            "error",
                        )
                        break

                print()

        finally:
            # Worktree mode: nothing to restore (Va never moves)
            # Legacy mode: restore HEAD
            if self._worktree_mgr is None:
                self._git_restore_head()

        # Aggregate per-submission artifacts into the canonical snapshots so
        # the website's sync-leaderboard-data workflow has fresh data to copy
        # into website/data/. Runs only when at least one group succeeded.
        if success > 0:
            self._print_section("Publishing Website Snapshots")
            self._publish_to_website_snapshots()
            # Upload snapshots to HF write side using --skip-aggregation
            # to preserve the local aggregation results.
            self._sync_to_hf_write_side()

        # Summary
        self._print_section("Summary")
        self._log(f"Success: {success}, Failed: {failed}")

        # Check for unpushed results
        try:
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.config.benchmark_repo, capture_output=True, text=True, timeout=10,
            )
            current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else self.config.benchmark_branch
            if current_branch == "HEAD":
                current_branch = self.config.benchmark_branch
            unpushed = subprocess.run(
                ["git", "log", "--oneline", f"origin/{current_branch}..HEAD"],
                cwd=self.config.benchmark_repo,
                capture_output=True, text=True, timeout=10,
            )
            if unpushed.returncode == 0 and unpushed.stdout.strip():
                unpushed_count = len(unpushed.stdout.strip().splitlines())
                self._log(
                    f"WARNING: {unpushed_count} commit(s) not yet pushed to remote!",
                    "error",
                )
                self._log(
                    f"Run:  cd {self.config.benchmark_repo} && "
                    f"git push origin {current_branch}",
                    "warn",
                )
                self._log(
                    "Results will NOT appear on the website until pushed.",
                    "warn",
                )
        except Exception:
            pass

        if self._log_fh:
            self._log(f"Full log saved to: {self._log_fh.name}")
            self._log_fh.close()
            self._log_fh = None

        return 0 if failed == 0 else 1

    def _try_upstream_lookup(self, group: CommitGroup) -> Optional[dict]:
        """Try to find existing upstream baseline for a group.

        Lookup order:
          1. Local submissions directory in benchmark repo
          2. Leaderboard snapshot (local or GitHub)
        """
        sha = group.latest_commit
        run_id = f"group-{group.group_id}-{sha[:12]}"

        # 1. Check local submissions
        result = self._baseline_lookup.lookup_by_submission_dir(
            self.config.benchmark_repo, run_id
        )
        if result:
            return result

        # 2. Check leaderboard snapshot (local or GitHub)
        result = self._baseline_lookup.lookup(sha)
        if result:
            return result

        self._log(f"No existing baseline found for {group.group_id}, will run benchmark")
        return None

    def _generate_report_from_checkpoint(self) -> None:
        """Generate attribution report from existing checkpoint."""
        self._log("Generating attribution report...")
        
        # Check if there's any data to report
        if not self.checkpoint.attribution and not self.checkpoint.upstream_baselines:
            self._log("No attribution data or upstream baselines found in checkpoint", "warn")
            return
        
        # Generate report
        generator = ReportGenerator(self.config)
        
        # Generate JSON report
        json_data = generator.generate_json(self.checkpoint)
        output_path = Path(self.config.attribution_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        self._log(f"JSON report saved to: {output_path}")
        
        # Generate HTML report
        html_path = Path(self.config.html_report_file)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        generator.generate_html(json_data, html_path)
        self._log(f"HTML report saved to: {html_path}")


# =============================================================================
# Report Generator
# =============================================================================

class ReportGenerator:
    """Generates attribution reports."""

    def __init__(self, config: Config):
        self.config = config

    def calculate_summary(self, checkpoint: Checkpoint) -> dict:
        """Calculate summary statistics."""
        attribution = checkpoint.attribution

        if not attribution:
            return {"total_contributions": 0}

        improvements = [a["improvement_percent"] for a in attribution]
        deltas = [a["delta_tps"] for a in attribution]

        # By member
        by_member: dict = {}
        for entry in attribution:
            m = entry["org_member"]
            if m not in by_member:
                by_member[m] = []
            by_member[m].append(entry["improvement_percent"])

        contributions_by_member = [
            {"member": m, "count": len(v), "avg_improvement": sum(v)/len(v), "total_improvement": sum(v)}
            for m, v in by_member.items()
        ]
        contributions_by_member.sort(key=lambda x: x["total_improvement"], reverse=True)

        # Top 5
        sorted_by_imp = sorted(attribution, key=lambda x: x.get("improvement_percent", 0), reverse=True)
        top_contributions = [
            {"org_member": e["org_member"], "improvement_percent": e["improvement_percent"], "group_id": e["group_id"]}
            for e in sorted_by_imp[:5]
        ]

        # Timeline
        timeline = sorted(attribution, key=lambda x: x.get("latest_commit_date", ""))

        return {
            "total_contributions": len(attribution),
            "avg_improvement_percent": round(sum(improvements)/len(improvements), 2) if improvements else 0,
            "max_improvement_percent": max(improvements) if improvements else 0,
            "min_improvement_percent": min(improvements) if improvements else 0,
            "total_delta_tps": round(sum(deltas), 2),
            "avg_delta_tps": round(sum(deltas)/len(deltas), 2) if deltas else 0,
            "contributions_by_member": contributions_by_member,
            "top_contributions": top_contributions,
            "timeline": timeline
        }

    def generate_json(self, checkpoint: Checkpoint) -> dict:
        """Generate JSON attribution file."""
        summary = self.calculate_summary(checkpoint)

        # Include upstream baselines in output
        upstream_baselines = []
        for sha, data in checkpoint.upstream_baselines.items():
            upstream_baselines.append({
                "commit_sha": sha,
                "throughput_tps": data.get("throughput_tps", 0),
                "ttft_ms": data.get("ttft_ms", 0),
                "timestamp": data.get("timestamp", ""),
                "source": data.get("source", "unknown"),
            })
        upstream_baselines.sort(key=lambda x: x.get("timestamp", ""))

        return {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model": self.config.model_name,
            "scenario": self.config.benchmark_scenario,
            "contributors": checkpoint.attribution,
            "upstream_baselines": upstream_baselines,
            "summary": summary
        }

    def generate_html(self, data: dict, output_path: Path) -> None:
        """Generate HTML report."""
        summary = data.get("summary", {})
        contributors = data.get("contributors", [])
        upstream_baselines = data.get("upstream_baselines", [])
        by_member = summary.get("contributions_by_member", [])

        max_improvement = max([m["avg_improvement"] for m in by_member]) if by_member else 100
        bar_scale = 100 / max(1, abs(max_improvement)) if max_improvement != 0 else 1

        timestamp = datetime.utcnow().isoformat() + "Z"
        model = data.get("model", "N/A")
        scenario = data.get("scenario", "N/A")

        # Build HTML using list concatenation to avoid f-string issues
        html_parts = []

        # HTML header
        html_parts.append('<!DOCTYPE html>')
        html_parts.append('<html lang="en">')
        html_parts.append('<head>')
        html_parts.append('    <meta charset="UTF-8">')
        html_parts.append('    <meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_parts.append('    <title>vLLM-HUST Contribution Attribution Report</title>')
        html_parts.append('    <style>')

        # CSS variables
        html_parts.append('        :root {')
        html_parts.append('            --primary: #667eea;')
        html_parts.append('            --primary-dark: #5a67d8;')
        html_parts.append('            --success: #22c55e;')
        html_parts.append('            --danger: #ef4444;')
        html_parts.append('            --bg: #f5f7fa;')
        html_parts.append('            --card-bg: #fff;')
        html_parts.append('            --text: #333;')
        html_parts.append('        }')
        html_parts.append('        * { box-sizing: border-box; margin: 0; padding: 0; }')
        html_parts.append('        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;')
        html_parts.append('               background: var(--bg); color: var(--text); line-height: 1.6; }')
        html_parts.append('        .container { max-width: 1400px; margin: 0 auto; padding: 24px; }')
        html_parts.append('        header { background: linear-gradient(135deg, var(--primary), var(--primary-dark));')
        html_parts.append('                   color: white; padding: 32px; border-radius: 16px; margin-bottom: 32px;')
        html_parts.append('                   box-shadow: 0 4px 20px rgba(102,126,234,.3); }')
        html_parts.append('        header h1 { font-size: 2em; margin-bottom: 8px; }')
        html_parts.append('        header .meta { opacity: 0.9; font-size: 0.9em; }')
        html_parts.append('        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));')
        html_parts.append('                        gap: 20px; margin-bottom: 32px; }')
        html_parts.append('        .stat-card { background: var(--card-bg); padding: 24px; border-radius: 12px;')
        html_parts.append('                       box-shadow: 0 2px 12px rgba(0,0,0,.08); }')
        html_parts.append('        .stat-card .label { color: #666; font-size: 0.8em; text-transform: uppercase; margin-bottom: 8px; }')
        html_parts.append('        .stat-card .value { font-size: 2.2em; font-weight: 700; }')
        html_parts.append('        .stat-card .value.positive { color: var(--success); }')
        html_parts.append('        .stat-card .value.negative { color: var(--danger); }')
        html_parts.append('        .section { background: var(--card-bg); border-radius: 12px; padding: 28px;')
        html_parts.append('                     margin-bottom: 28px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }')
        html_parts.append('        .section h2 { font-size: 1.25em; margin-bottom: 20px; border-bottom: 2px solid var(--primary); padding-bottom: 12px; }')
        html_parts.append('        table { width: 100%; border-collapse: collapse; }')
        html_parts.append('        th { background: #f8fafc; text-align: left; padding: 14px 16px; font-weight: 600;')
        html_parts.append('              color: #666; border-bottom: 2px solid #e5e7eb; font-size: 0.85em; text-transform: uppercase; }')
        html_parts.append('        td { padding: 14px 16px; border-bottom: 1px solid #f1f5f9; }')
        html_parts.append('        tr:hover { background: #f8fafc; }')
        html_parts.append('        .tag { display: inline-block; padding: 4px 12px; border-radius: 20px;')
        html_parts.append('                font-size: 0.8em; font-weight: 600; }')
        html_parts.append('        .tag-pr { background: #dbeafe; color: #1d4ed8; }')
        html_parts.append('        .tag-session { background: #dcfce7; color: #166534; }')
        html_parts.append('        .tag-upstream { background: #fef3c7; color: #92400e; }')
        html_parts.append('        .improvement { font-weight: 700; }')
        html_parts.append('        .improvement.positive { color: var(--success); }')
        html_parts.append('        .improvement.negative { color: var(--danger); }')
        html_parts.append('        .bar-chart { display: flex; flex-direction: column; gap: 14px; }')
        html_parts.append('        .bar-row { display: flex; align-items: center; gap: 16px; }')
        html_parts.append('        .bar-label { min-width: 130px; font-weight: 600; text-align: right; }')
        html_parts.append('        .bar-container { flex: 1; background: #e5e7eb; border-radius: 10px;')
        html_parts.append('                          height: 36px; overflow: hidden; }')
        html_parts.append('        .bar-fill { height: 100%; border-radius: 10px; display: flex; align-items: center;')
        html_parts.append('                      padding: 0 12px; min-width: 60px; transition: width 0.6s; }')
        html_parts.append('        .bar-fill.positive { background: linear-gradient(90deg, #22c55e, #4ade80); color: white; }')
        html_parts.append('        .bar-fill.negative { background: linear-gradient(90deg, #ef4444, #f87171); color: white; }')
        html_parts.append('        .bar-value { min-width: 90px; text-align: right; font-weight: 700; font-size: 1.1em; }')
        html_parts.append('        .bar-value.positive { color: var(--success); }')
        html_parts.append('        .bar-value.negative { color: var(--danger); }')
        html_parts.append('        .highlight-box { background: linear-gradient(135deg, #fef3c7, #fde68a);')
        html_parts.append('                          border-left: 4px solid #f59e0b; padding: 18px 20px;')
        html_parts.append('                          border-radius: 10px; margin-bottom: 24px; }')
        html_parts.append('        .footer { text-align: center; color: #666; padding: 24px; font-size: 0.85em; }')
        html_parts.append('        a { color: var(--primary); text-decoration: none; }')
        html_parts.append('        a:hover { text-decoration: underline; }')
        html_parts.append('        @media (max-width: 768px) { .stats-grid { grid-template-columns: 1fr 1fr; } }')
        html_parts.append('    </style>')
        html_parts.append('</head>')
        html_parts.append('<body>')
        html_parts.append('    <div class="container">')
        html_parts.append('        <header>')
        html_parts.append('            <h1>vLLM-HUST Contribution Attribution Report</h1>')
        html_parts.append(f'            <div class="meta">Generated: {timestamp} | Model: {model} | Scenario: {scenario}</div>')
        html_parts.append('        </header>')
        html_parts.append('        <div class="highlight-box">')
        html_parts.append('            <strong>Attribution Model:</strong> Each org member\'s contribution is measured as the performance delta')
        html_parts.append('            against the previous upstream benchmark result. Only org member results appear on leaderboard.')
        html_parts.append('        </div>')
        html_parts.append('        <div class="stats-grid">')

        # Stat cards
        total = summary.get('total_contributions', 0)
        avg_imp = summary.get('avg_improvement_percent', 0)
        total_delta = summary.get('total_delta_tps', 0)
        max_imp = summary.get('max_improvement_percent', 0)

        html_parts.append('            <div class="stat-card">')
        html_parts.append('                <div class="label">Total Contributions</div>')
        html_parts.append(f'                <div class="value">{total}</div>')
        html_parts.append('            </div>')

        avg_class = 'positive' if avg_imp >= 0 else 'negative'
        avg_sign = '+' if avg_imp >= 0 else ''
        html_parts.append('            <div class="stat-card">')
        html_parts.append('                <div class="label">Avg Throughput Improvement</div>')
        html_parts.append(f'                <div class="value {avg_class}">{avg_sign}{avg_imp:.2f}%</div>')
        html_parts.append('            </div>')

        delta_class = 'positive' if total_delta >= 0 else 'negative'
        delta_sign = '+' if total_delta >= 0 else ''
        html_parts.append('            <div class="stat-card">')
        html_parts.append('                <div class="label">Total TPS Gain</div>')
        html_parts.append(f'                <div class="value {delta_class}">{delta_sign}{total_delta:.2f}</div>')
        html_parts.append('            </div>')

        max_class = 'positive' if max_imp >= 0 else 'negative'
        max_sign = '+' if max_imp >= 0 else ''
        html_parts.append('            <div class="stat-card">')
        html_parts.append('                <div class="label">Max Improvement</div>')
        html_parts.append(f'                <div class="value {max_class}">{max_sign}{max_imp:.2f}%</div>')
        html_parts.append('            </div>')
        html_parts.append('        </div>')

        # Improvement by member chart
        html_parts.append('        <div class="section">')
        html_parts.append('            <h2>Improvement by Member</h2>')
        html_parts.append('            <div class="bar-chart">')

        for m in by_member:
            improv = m["avg_improvement"]
            bar_width = min(abs(improv) * bar_scale, 100) if bar_scale > 0 else 0
            bar_width = max(bar_width, 1)
            color_class = "positive" if improv >= 0 else "negative"
            sign = '+' if improv >= 0 else ''
            html_parts.append('                <div class="bar-row">')
            html_parts.append(f'                    <div class="bar-label">{m["member"]}</div>')
            html_parts.append(f'                    <div class="bar-container">')
            html_parts.append(f'                        <div class="bar-fill {color_class}" style="width: {bar_width:.1f}%">{m["count"]} contrib</div>')
            html_parts.append('                    </div>')
            html_parts.append(f'                    <div class="bar-value {color_class}">{sign}{improv:.2f}%</div>')
            html_parts.append('                </div>')

        html_parts.append('            </div>')
        html_parts.append('        </div>')

        # Upstream baselines table (leaderboard reference)
        if upstream_baselines:
            html_parts.append('        <div class="section">')
            html_parts.append('            <h2>Upstream Baselines (Leaderboard Reference)</h2>')
            html_parts.append('            <p style="color:#666;margin-bottom:16px;">These upstream results serve as comparison baselines. '
                              'They are shown on the leaderboard for reference but are not attributed as org contributions.</p>')
            html_parts.append('            <table>')
            html_parts.append('                <thead>')
            html_parts.append('                    <tr>')
            html_parts.append('                        <th>Commit</th><th>Type</th><th>Date</th>')
            html_parts.append('                        <th>Throughput (TPS)</th><th>TTFT (ms)</th><th>Source</th>')
            html_parts.append('                    </tr>')
            html_parts.append('                </thead>')
            html_parts.append('                <tbody>')

            for ub in sorted(upstream_baselines, key=lambda x: x.get("timestamp", "")):
                sha = ub.get("commit_sha", "N/A")
                sha_short = sha[:12] if sha else "N/A"
                tps = ub.get("throughput_tps", 0)
                ttft = ub.get("ttft_ms", 0)
                ts = ub.get("timestamp", "N/A")[:10]
                source = ub.get("source", "unknown")
                source_label = "Lookup" if source == "leaderboard_lookup" else ("Local" if source == "local_submission" else "Run")

                html_parts.append('                    <tr>')
                html_parts.append(f'                        <td><code>{sha_short}</code></td>')
                html_parts.append(f'                        <td><span class="tag tag-upstream">upstream</span></td>')
                html_parts.append(f'                        <td>{ts}</td>')
                html_parts.append(f'                        <td><strong>{tps:.2f}</strong></td>')
                html_parts.append(f'                        <td>{ttft:.2f}</td>')
                html_parts.append(f'                        <td>{source_label}</td>')
                html_parts.append('                    </tr>')

            html_parts.append('                </tbody>')
            html_parts.append('            </table>')
            html_parts.append('        </div>')

        # All contributions table
        html_parts.append('        <div class="section">')
        html_parts.append('            <h2>All Contributions</h2>')
        html_parts.append('            <table>')
        html_parts.append('                <thead>')
        html_parts.append('                    <tr>')
        html_parts.append('                        <th>Member</th><th>Type</th><th>Group</th><th>Date</th>')
        html_parts.append('                        <th>Baseline TPS</th><th>Result TPS</th><th>Delta TPS</th><th>Improvement</th>')
        html_parts.append('                    </tr>')
        html_parts.append('                </thead>')
        html_parts.append('                <tbody>')

        for c in sorted(contributors, key=lambda x: x.get("latest_commit_date", "")):
            improv = c.get("improvement_percent", 0)
            delta = c.get("delta_tps", 0)
            upstream = c.get("upstream_throughput_tps", 0)
            org_tps = c.get("org_throughput_tps", 0)
            date_str = c.get("latest_commit_date", "N/A")[:10]
            color_class = "positive" if improv >= 0 else "negative"
            sign = '+' if improv >= 0 else ''

            html_parts.append('                    <tr>')
            html_parts.append(f'                        <td><strong>{c.get("org_member", "N/A")}</strong></td>')
            html_parts.append(f'                        <td><span class="tag tag-{c.get("group_type", "session")}">{c.get("group_type", "session")}</span></td>')
            html_parts.append(f'                        <td>{c.get("group_id", "N/A")}</td>')
            html_parts.append(f'                        <td>{date_str}</td>')
            html_parts.append(f'                        <td>{upstream:.2f}</td>')
            html_parts.append(f'                        <td>{org_tps:.2f}</td>')
            html_parts.append(f'                        <td>{sign}{delta:.2f}</td>')
            html_parts.append(f'                        <td><span class="improvement {color_class}">{sign}{improv:.2f}%</span></td>')
            html_parts.append('                    </tr>')

        html_parts.append('                </tbody>')
        html_parts.append('            </table>')
        html_parts.append('        </div>')

        # Footer
        html_parts.append('        <div class="footer">')
        html_parts.append('            Generated by vllm-hust-benchmark | <a href="https://vllm-hust.sage.org.ai/">Leaderboard</a>')
        html_parts.append('        </div>')
        html_parts.append('    </div>')
        html_parts.append('</body>')
        html_parts.append('</html>')

        with open(output_path, "w") as f:
            f.write('\n'.join(html_parts))

    def run(self) -> int:
        """Generate reports from checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_file)
        if not checkpoint_path.exists():
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}", file=sys.stderr)
            return 1

        checkpoint = Checkpoint.from_file(checkpoint_path)
        data = self.generate_json(checkpoint)

        # Write JSON
        output_path = Path(self.config.attribution_file)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"JSON report saved: {output_path}")

        # Write HTML
        html_path = Path(self.config.html_report_file)
        self.generate_html(data, html_path)
        print(f"HTML report saved: {html_path}")

        return 0


# =============================================================================
# CLI Entry Point
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="vLLM-HUST Org Member Benchmark Runner with Delta Attribution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Directory Layout:
              The script auto-detects repos as sibling directories:
                workspace/
                ├── vllm-hust/               <- Va stable vllm-hust (worktree mode)
                ├── vllm-ascend-hust/        <- Va stable vllm-ascend-hust (worktree mode)
                ├── vllm-hust-benchmark/     <- Auto-detected if sibling exists
                └── hust-tools/
                    └── run_org_member_benchmarks.py

            Worktree Mode (Va/Vb Isolation):
              When --vllm-hust-repo and --vllm-ascend-hust-repo are provided:
                Va (stable): hosts CI scripts, constraints files, submission output
                Vb (ephemeral): per-commit worktree, created before each run, deleted after
              At startup, stale worktrees from crashed runs are auto-cleaned.

            Timeline Cache:
              Commit timeline is cached to .benchmarks/commit-timeline-cache.json (24h TTL).
              Use --refresh-timeline to force a fresh GitHub API fetch.
              Use --timeline-cache-ttl to customize the TTL in seconds.

            Cross-Repo Commit Resolution:
              For each vllm-hust commit, the corresponding vllm-ascend-hust commit is resolved:
                1. PR merge SHA lookup in vllm-ascend-hust (GitHub API)
                2. Time-based fallback: git log --before=<date> on origin/main
              If the PR SHA is not fetchable locally, the time-based SHA is used directly.

            Run Command Arguments:
              --dry-run               Simulate without running benchmarks
              --resume                Resume from checkpoint
              --vllm-hust-repo PATH   Va stable vllm-hust worktree (enables worktree mode)
              --vllm-ascend-hust-repo PATH  Va stable vllm-ascend-hust worktree
              --benchmark-repo PATH   vllm-hust-benchmark repo (auto-detected if sibling)
              --scenario NAME         Benchmark scenario (default: random-online)
              --model NAME            Model name (default: Qwen/Qwen2.5-14B-Instruct)
              --chips N               Number of chips (default: 1)
              --since DATE            Start date YYYY-MM-DD (default: 2026-01-01)
              --exclude MEMBERS       Comma-separated excluded members
              --checkpoint PATH       Checkpoint file path
              --attribution PATH      Attribution JSON output file
              --html PATH             HTML report output file
              --branch NAME           GitHub branch (default: main)
              --include-upstream 0|1  Run upstream benchmarks (default: 1)
              --upstream-commits SHAs Comma-separated upstream commit SHAs for baseline
              --baseline-lookup PATH  Path for existing baseline lookup ('no-lookup' to disable)
              --log-file PATH         Tee all output to this file
              --refresh-timeline      Force-refresh commit timeline from GitHub
              --timeline-cache-ttl N  Cache TTL in seconds (default: 86400 = 24h)

            Report Command Arguments:
              --checkpoint PATH       Checkpoint file path
              --output PATH           JSON output file
              --html PATH             HTML output file
              --model NAME            Model name filter
              --scenario NAME         Scenario filter

            Examples:
                # Run with worktree mode (recommended)
                GH_TOKEN=ghp_xxx python3 run_org_member_benchmarks.py run \\
                    --vllm-hust-repo /workspace/vllm-hust \\
                    --vllm-ascend-hust-repo /workspace/vllm-ascend-hust

                # Dry run (simulate without benchmarking)
                GH_TOKEN=ghp_xxx python3 run_org_member_benchmarks.py run --dry-run

                # Force-refresh timeline from GitHub
                GH_TOKEN=ghp_xxx python3 run_org_member_benchmarks.py run \\
                    --vllm-hust-repo /workspace/vllm-hust \\
                    --vllm-ascend-hust-repo /workspace/vllm-ascend-hust \\
                    --refresh-timeline

                # Resume from checkpoint
                GH_TOKEN=ghp_xxx python3 run_org_member_benchmarks.py run \\
                    --resume --model Qwen/Qwen2.5-7B-Instruct

                # Generate reports only
                python3 run_org_member_benchmarks.py report \\
                    --checkpoint .benchmarks/checkpoint.json
        """)
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks for org members")
    run_parser.add_argument("--dry-run", action="store_true", help="Simulate without pushing")
    run_parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    run_parser.add_argument("--fail-fast", action="store_true", help="Exit on first group failure (for debugging)")
    run_parser.add_argument("--vllm-hust-repo", help="Path to Va stable vllm-hust worktree (enables worktree mode)")
    run_parser.add_argument("--vllm-ascend-hust-repo", help="Path to Va stable vllm-ascend-hust worktree (enables worktree mode)")
    run_parser.add_argument("--benchmark-repo", help="Path to vllm-hust-benchmark repo (auto-detected if sibling)")
    run_parser.add_argument("--scenario", default="random-online", help="Benchmark scenario")
    run_parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct", help="Model name")
    run_parser.add_argument("--chips", type=int, default=1, help="Number of chips")
    run_parser.add_argument("--since", default="2026-01-01", help="Start date (YYYY-MM-DD)")
    run_parser.add_argument("--exclude", default="ShuhaoZhangTony,moonandlife", help="Excluded members")
    run_parser.add_argument("--checkpoint", help="Checkpoint file path")
    run_parser.add_argument("--attribution", help="Attribution output file")
    run_parser.add_argument("--branch", default="main", help="GitHub branch")
    run_parser.add_argument("--include-upstream", type=int, default=1, help="Run upstream benchmarks (1=yes, 0=no)")
    run_parser.add_argument("--upstream-commits", default="",
                            help="Comma-separated upstream commit SHAs for baseline comparison")
    run_parser.add_argument("--baseline-lookup", default="",
                            help="Path to benchmark repo for existing baseline lookup (default: auto-detect, 'no-lookup' to disable)")
    run_parser.add_argument("--log-file", default="", help="Log all output to this file (tee: terminal + file)")
    run_parser.add_argument("--refresh-timeline", action="store_true",
                            help="Force-refresh commit timeline from GitHub (bypass cache)")
    run_parser.add_argument("--timeline-cache-ttl", type=int, default=86400,
                            help="Cache TTL in seconds (default: 86400 = 24h)")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate attribution reports")
    report_parser.add_argument("--checkpoint", help="Checkpoint file path")
    report_parser.add_argument("--output", help="JSON output file")
    report_parser.add_argument("--html", help="HTML output file")
    report_parser.add_argument("--model", help="Model name")
    report_parser.add_argument("--scenario", help="Benchmark scenario")

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Get GitHub token
    github_token = os.environ.get("GH_TOKEN", "")
    if args.command == "run" and not github_token:
        print("[ERROR] GH_TOKEN environment variable is required for 'run' command", file=sys.stderr)
        return 1

    # Create config - handle None values from argparse
    script_dir = Path(__file__).parent.resolve()

    # Helper to get arg or default, handling None values from argparse
    def get_arg(arg_attr: str, default_val: Any) -> Any:
        val = getattr(args, arg_attr, None)
        return default_val if val is None else val

    # Benchmark repo: use explicit arg, else look for sibling vllm-hust-benchmark dir, else use script_dir
    benchmark_repo_arg = get_arg("benchmark_repo", "")
    if benchmark_repo_arg:
        benchmark_repo = Path(benchmark_repo_arg).resolve()
    else:
        # Look for sibling directory vllm-hust-benchmark.
        # script_dir is the scripts/ subdirectory, so:
        #   benchmark_repo  = script_dir            (when scripts/ is inside vllm-hust-benchmark/)
        #   benchmark_repo  = script_dir.parent      (when scripts/ is the repo root itself)
        # We check both: scripts/ parent IS the repo, or scripts/ parent/parent IS the repo.
        candidate = script_dir  # scripts/ subdirectory
        sibling = candidate.parent / "vllm-hust-benchmark"
        if not (sibling.exists() and sibling.is_dir()):
            # scripts/ is inside vllm-hust-benchmark/ (normal layout)
            candidate = script_dir.parent  # scripts/ -> vllm-hust-benchmark/
        sibling = candidate / "vllm-hust-benchmark"
        if sibling.exists() and sibling.is_dir():
            benchmark_repo = sibling
        else:
            benchmark_repo = candidate

    # Va (stable infrastructure): vllm-hust worktree
    vllm_hust_repo_arg = get_arg("vllm_hust_repo", "")
    if vllm_hust_repo_arg:
        vllm_hust_repo = Path(vllm_hust_repo_arg).resolve()
    else:
        vllm_hust_sibling = script_dir.parent / "vllm-hust"
        vllm_hust_repo = vllm_hust_sibling if vllm_hust_sibling.exists() and vllm_hust_sibling.is_dir() else None

    # Va (stable infrastructure): vllm-ascend-hust worktree
    vllm_ascend_hust_repo_arg = get_arg("vllm_ascend_hust_repo", "")
    if vllm_ascend_hust_repo_arg:
        vllm_ascend_hust_repo = Path(vllm_ascend_hust_repo_arg).resolve()
    else:
        ascend_sibling = script_dir.parent / "vllm-ascend-hust"
        vllm_ascend_hust_repo = ascend_sibling if ascend_sibling.exists() and ascend_sibling.is_dir() else None

    config = Config(
        excluded_members=get_arg("exclude", "ShuhaoZhangTony,moonandlife"),
        benchmark_scenario=get_arg("scenario", "random-online"),
        model_name=get_arg("model", "Qwen/Qwen2.5-14B-Instruct"),
        chip_count=get_arg("chips", 1),
        since_date=get_arg("since", "2026-01-01"),
        benchmark_branch=get_arg("branch", "main"),
        checkpoint_file=get_arg("checkpoint", ".benchmarks/org-member-benchmarks/checkpoint.json"),
        attribution_file=get_arg("output", "contribution-attribution.json"),
        html_report_file=get_arg("html", "contribution-report.html"),
        dry_run=get_arg("dry_run", False),
        resume_mode=get_arg("resume", False),
        fail_fast=get_arg("fail_fast", False),
        include_upstream=bool(get_arg("include_upstream", True)),
        upstream_commits=get_arg("upstream_commits", ""),
        baseline_lookup=get_arg("baseline_lookup", ""),
        script_dir=script_dir,
        benchmark_repo=benchmark_repo,
        vllm_hust_repo=vllm_hust_repo,
        vllm_ascend_hust_repo=vllm_ascend_hust_repo,
        timeline_cache_file=".benchmarks/commit-timeline-cache.json",
        refresh_timeline=get_arg("refresh_timeline", False),
        timeline_cache_ttl=get_arg("timeline_cache_ttl", 86400),
    )

    if args.command == "run":
        log_file = getattr(args, "log_file", "") or ""
        runner = BenchmarkRunner(config, github_token, log_file=log_file)
        return runner.run()
    elif args.command == "report":
        reporter = ReportGenerator(config)
        return reporter.run()

    return 0


if __name__ == "__main__":
    sys.exit(main())