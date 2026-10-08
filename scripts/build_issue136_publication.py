#!/usr/bin/env python3
"""Build canonical Issue #136 snapshot candidates from completed archives."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm_hust_benchmark.issue136_publication import (
    PublicationError,
    build_publication_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-archive", type=Path, required=True)
    parser.add_argument("--scaled-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = build_publication_bundle(
            args.fixed_archive, args.scaled_archive, args.output_dir
        )
    except (PublicationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
