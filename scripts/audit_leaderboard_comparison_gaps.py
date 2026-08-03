#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vllm_hust_benchmark.comparison_gap_audit import (  # noqa: E402
    build_comparison_gap_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--current-core-head")
    parser.add_argument("--current-plugin-head")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Return 2 unless every in-scope target has a strict matched pair.",
    )
    args = parser.parse_args()
    report = build_comparison_gap_audit(
        args.repo_root.resolve(),
        current_core_head=args.current_core_head,
        current_plugin_head=args.current_plugin_head,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    summary = report["summary"]
    print(
        "comparison gap audit: "
        f"targets={summary['target_count']} "
        f"ready={summary['ready_pair_count']} "
        f"rerun_targets={summary['rerun_target_count']} "
        f"rerun_jobs={summary['rerun_job_count']}",
        file=sys.stderr,
    )
    if args.require_complete and summary["ready_pair_count"] != summary["target_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
