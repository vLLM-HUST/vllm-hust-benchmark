#!/usr/bin/env python3
"""Validate a trend entry JSON file or directory for CI and local use."""

from __future__ import annotations

import argparse
from pathlib import Path

from vllm_hust_benchmark.trend_validator import load_json_entries, validate_entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="JSON file or directory of JSON files"
    )
    args = parser.parse_args()
    paths = [args.input] if args.input.is_file() else sorted(args.input.glob("*.json"))
    entries = []
    for path in paths:
        entries.extend(load_json_entries(path))
    report = validate_entries(entries)
    for decision in report.decisions:
        print(f"{decision.status:12} {decision.entry_id}: {decision.reason}")
    for issue in report.issues:
        print(
            f"{issue.severity.upper():5} {issue.code}: {issue.entry_id}: {issue.message}"
        )
    return 1 if not report.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
