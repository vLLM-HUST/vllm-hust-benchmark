#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_hust_benchmark.snapshot_target_binding import (
    bind_snapshot_set,
    load_official_target_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strictly bind leaderboard snapshots to the official target registry."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--snapshot-dir", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    snapshot_dir = (
        args.snapshot_dir.resolve()
        if args.snapshot_dir
        else repo_root / "leaderboard-data" / "snapshots"
    )
    registry = load_official_target_registry(repo_root)
    report = bind_snapshot_set(snapshot_dir, registry)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
