#!/usr/bin/env python3
"""Capture immutable and working-tree provenance for an official source checkout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import subprocess
from pathlib import Path


def _run(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tree_digest(repo: Path) -> tuple[str, int, int, int]:
    paths = set(
        path
        for path in _run(repo, "ls-files", "-co", "--exclude-standard", "-z").split(
            b"\0"
        )
        if path
    )
    tracked_paths = set(paths)
    generated_paths = {
        b"vllm/_version.py",
        b"vllm_ascend/_build_info.py",
        b"vllm_ascend/_version.py",
    }
    generated_paths.update(
        os.fsencode(path.relative_to(repo))
        for pattern in ("vllm-*.dist-info/*", "vllm_ascend-*.dist-info/*")
        for path in repo.glob(pattern)
        if path.is_file()
    )
    generated_paths.update(
        os.fsencode(path.relative_to(repo))
        for package_dir in (repo / "vllm", repo / "vllm_ascend")
        if package_dir.is_dir()
        for path in package_dir.rglob("*")
        if path.is_file()
        and path.name.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
    )
    paths.update(
        path for path in generated_paths if (repo / os.fsdecode(path)).is_file()
    )
    entries: list[bytes] = []
    tracked = 0
    untracked = 0
    generated = 0
    for raw_path in sorted(paths):
        path = repo / os.fsdecode(raw_path)
        if not path.is_file():
            continue
        if b"\0" in raw_path:
            raise ValueError("source path contains NUL")
        if raw_path in tracked_paths:
            tracked += 1
        else:
            untracked += 1
            if raw_path in generated_paths:
                generated += 1
        digest = hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii")
        entries.append(raw_path + b"\t" + digest + b"\n")
    return _sha256(b"".join(entries)), tracked, untracked, generated


def capture(repo: Path, requested_ref: str, repository: str) -> dict[str, object]:
    if not (repo / ".git").exists() and not (repo / "HEAD").exists():
        raise ValueError(f"not a git worktree: {repo}")
    observed_commit = (
        _run(repo, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    )
    patch = _run(repo, "diff", "--binary", "HEAD")
    status = _run(repo, "status", "--porcelain=v1", "--untracked-files=all")
    tree_digest, tracked_count, untracked_count, generated_count = _tree_digest(repo)
    return {
        "schema_version": "official-source-provenance/v1",
        "repository": repository,
        "requested_ref": requested_ref,
        "observed_commit": observed_commit,
        "tracked_patch_sha256": _sha256(patch),
        "working_tree_sha256": tree_digest,
        "status": "clean" if not status and generated_count == 0 else "modified",
        "tracked_file_count": tracked_count,
        "untracked_file_count": untracked_count,
        "generated_file_count": generated_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("requested_ref")
    parser.add_argument("repository")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = capture(args.repo, args.requested_ref, args.repository)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
