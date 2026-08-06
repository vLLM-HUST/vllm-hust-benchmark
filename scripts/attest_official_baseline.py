#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_hust_benchmark.official_baseline_attestation import (  # noqa: E402
    attest_completed_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Attest one completed exact-target official baseline."
    )
    parser.add_argument("--staged-submission", type=Path, required=True)
    parser.add_argument("--result-spec-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verified-by", required=True)
    parser.add_argument("--minimum-repeats", type=int, default=3)
    parser.add_argument(
        "--comparison-side",
        choices=("baseline", "current"),
        default="baseline",
        help="Attest the official upstream baseline or its exact-spec current pair.",
    )
    args = parser.parse_args()
    entry = attest_completed_baseline(
        REPO_ROOT,
        args.staged_submission.resolve(),
        args.result_spec_dir.resolve(),
        args.output_dir.resolve(),
        verified_by=args.verified_by,
        minimum_repeats=args.minimum_repeats,
        comparison_side=args.comparison_side,
    )
    print(args.output_dir.resolve())
    print(entry["same_spec"]["spec_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
