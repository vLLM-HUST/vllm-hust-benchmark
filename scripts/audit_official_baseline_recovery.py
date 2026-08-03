#!/usr/bin/env python3
"""Audit legacy official baselines against the current fixed-target contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vllm_hust_benchmark.baseline_recovery import build_recovery_audit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-recoverable",
        action="store_true",
        help="Return 2 unless at least one active public baseline is recoverable.",
    )
    args = parser.parse_args()
    report = build_recovery_audit(args.repo_root.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    summary = report["summary"]
    print(
        "baseline recovery audit: "
        f"scanned={summary['scanned']} "
        f"active_public={summary['active_public_candidates']} "
        f"recoverable={summary['recoverable']} "
        f"rerun_required={summary['rerun_required']}",
        file=sys.stderr,
    )
    if args.require_recoverable and not summary["recoverable"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
