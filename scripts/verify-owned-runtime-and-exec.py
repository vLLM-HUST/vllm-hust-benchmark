#!/usr/bin/env python3
"""Fail closed on actual owned-container runtime identity, then exec the runner."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import sys


SCHEMA = "strict-owned-runtime-preflight/v1"
SOURCE_PATHS = {
    "core": Path("/vllm-workspace/vllm"),
    "backend": Path("/vllm-workspace/vllm-ascend"),
}


def git_identity(path: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {"path": str(path), "commit": head, "clean": not status.strip()}


def package_versions(names: list[str]) -> dict[str, str]:
    resolved = {}
    for name in names:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError as error:
            raise RuntimeError(
                f"required runtime package is missing: {name}"
            ) from error
    return resolved


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise RuntimeError("owned runtime preflight lacks a runner command")
    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    actual = {
        "schema_version": SCHEMA,
        "sources": {name: git_identity(path) for name, path in SOURCE_PATHS.items()},
        "packages": package_versions(sorted(expected["packages"])),
    }
    atomic_json(args.output, actual)
    for name, commit_field in (("core", "core_commit"), ("backend", "backend_commit")):
        source = actual["sources"][name]
        if not source["clean"] or source["commit"] != expected[commit_field]:
            raise RuntimeError(f"owned runtime {name} source identity mismatch")
    if actual["packages"] != expected["packages"]:
        raise RuntimeError("owned runtime package versions mismatch")
    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"owned runtime preflight rejected: {error}", file=sys.stderr)
        raise SystemExit(2) from error
