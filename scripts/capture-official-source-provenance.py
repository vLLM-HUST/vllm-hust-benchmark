#!/usr/bin/env python3
"""Capture immutable and working-tree provenance for an official source checkout."""

from __future__ import annotations

import argparse
import hashlib
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


def _tree_digest(repo: Path) -> tuple[str, int, int]:
    paths = _run(repo, "ls-files", "-co", "--exclude-standard", "-z").split(b"\0")
    entries: list[bytes] = []
    tracked = 0
    untracked = 0
    for raw_path in sorted(path for path in paths if path):
        path = repo / os.fsdecode(raw_path)
        if not path.is_file():
            continue
        if b"\0" in raw_path:
            raise ValueError("source path contains NUL")
        status = _run(repo, "ls-files", "--stage", "--", os.fsdecode(raw_path))
        if status:
            tracked += 1
        else:
            untracked += 1
        digest = hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii")
        entries.append(raw_path + b"\t" + digest + b"\n")
    return _sha256(b"".join(entries)), tracked, untracked


def capture(repo: Path, requested_ref: str, repository: str) -> dict[str, object]:
    if not (repo / ".git").exists() and not (repo / "HEAD").exists():
        raise ValueError(f"not a git worktree: {repo}")
    observed_commit = (
        _run(repo, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    )
    patch = _run(repo, "diff", "--binary", "HEAD")
    status = _run(repo, "status", "--porcelain=v1", "--untracked-files=all")
    tree_digest, tracked_count, untracked_count = _tree_digest(repo)
    return {
        "schema_version": "official-source-provenance/v1",
        "repository": repository,
        "requested_ref": requested_ref,
        "observed_commit": observed_commit,
        "tracked_patch_sha256": _sha256(patch),
        "working_tree_sha256": tree_digest,
        "status": "clean" if not status else "modified",
        "tracked_file_count": tracked_count,
        "untracked_file_count": untracked_count,
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
