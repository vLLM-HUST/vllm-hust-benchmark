#!/usr/bin/env python3
"""Audit existing public records from the pinned benchmark snapshot without running models."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

CHECKS = (
    (
        "reports/c16-baseline018-reuse-20261022/audit.py",
        "bf88e94a411f6f342cff5a962abab8d3cf676bdcf1a8c91d5397b4e7d72158c3",
        (),
    ),
    (
        "reports/frontier-raw-recovery-20261020/verify.py",
        "4dd62d6687439a025ec6a9420579d44fcaef8842cdf1be2c166e69edf073b613",
        (),
    ),
    (
        "scripts/recompute_b0_baseline_metrics.py",
        "c1c7e40426725a4c3a140178f7cdcdab3d36b583c22264107ea527c85e124565",
        ("--check",),
    ),
    (
        "reports/frontierscience-qwen35-b0-precheck-20261013/verify_public.py",
        "18499927f5d5ad4997fff5c8592d2a8863093dc63e1d96cee8c9ca676306f40d",
        (),
    ),
    (
        "reports/hle-verified-qwen35-20261015/verify_public.py",
        "44951c5ec481ad7b22d9d9b96014dd4ec6d8ab54b69aeed59a95222ec8cd6c82",
        (),
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Benchmark snapshot d67aea6eb3eb0dee271a01b4d66acce8b65bd034",
    )
    args = parser.parse_args()
    root = args.benchmark.resolve()
    for relative, expected, extra in CHECKS:
        script = root / relative
        if (
            not script.is_file()
            or hashlib.sha256(script.read_bytes()).hexdigest() != expected
        ):
            parser.exit(2, f"Missing or unexpected verifier: {script}\n")
    for relative, expected, extra in CHECKS:
        print(f"\n--- {relative}", flush=True)
        result = subprocess.run(
            [sys.executable, "-B", str(root / relative), *extra], cwd=root
        )
        if result.returncode:
            return result.returncode
    print("\nVerified five public archive checks. No new model calls.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
