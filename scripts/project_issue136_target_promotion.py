#!/usr/bin/env python3
"""Project, but do not apply, Issue #136 evidence-backed target promotion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm_hust_benchmark.issue136_target_promotion import (
    PromotionError,
    write_promotion_projection,
)
from vllm_hust_benchmark.snapshot_target_binding import load_official_target_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        registry = load_official_target_registry(args.repo_root.resolve())
        output = write_promotion_projection(args.bundle, registry, args.output_dir)
    except (PromotionError, OSError, ValueError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
