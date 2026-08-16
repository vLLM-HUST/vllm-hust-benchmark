#!/usr/bin/env python3
# ruff: noqa: I001
"""Build the separate, auditable historical leaderboard projection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_hust_benchmark.historical_recovery import write_recovery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover usable historical entries without changing formal admission."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("leaderboard-data/snapshots"),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("src/vllm_hust_benchmark/data/official_targets.json"),
    )
    parser.add_argument(
        "--revision-aliases",
        type=Path,
        default=Path("leaderboard-data/historical-revision-aliases.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    registry = args.registry
    if not registry.is_absolute():
        registry = repo_root / registry
    revision_aliases = args.revision_aliases
    if not revision_aliases.is_absolute():
        revision_aliases = repo_root / revision_aliases
    entries_path, report_path = write_recovery(
        repo_root=repo_root,
        output_dir=output_dir,
        registry_path=registry,
        revision_aliases_path=revision_aliases,
    )
    print(entries_path)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
