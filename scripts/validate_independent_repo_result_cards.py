#!/usr/bin/env python3
"""Validate the independent optimization repo result card registry.

Issue #89: the independent_repo_registry module, data file, schema and
tests are self-consistent, but without a CI consumer the registry is
only validated on demand.  This script wires load_registry into the
CI validate job so that a stale, incomplete or schema-invalid registry
fails the build (fail-closed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vllm_hust_benchmark.independent_repo_registry import load_registry  # noqa: E402

DEFAULT_REGISTRY = (
    REPO_ROOT
    / "leaderboard-data"
    / "independent-repos"
    / "independent_repo_result_cards.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help=f"Path to the result card registry JSON (default: {DEFAULT_REGISTRY})",
    )
    args = parser.parse_args()

    registry_path: Path = args.registry
    if not registry_path.is_file():
        print(
            f"ERROR: independent repo result card registry not found: {registry_path}",
            file=sys.stderr,
        )
        return 2

    try:
        load_registry(registry_path)
    except ValueError as exc:
        print(
            f"ERROR: independent repo result card registry validation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print(f"OK: independent repo result card registry validated: {registry_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
