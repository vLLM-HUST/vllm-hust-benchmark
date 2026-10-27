"""Prepare pinned AllenAI NarrativeQA metric source, not model execution."""

import argparse
import json
from pathlib import Path
import sys

from prepare_xsum_original_assets import main as acquire

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/narrativeqa-metric-pins.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pins = json.loads(PINS.read_text("utf-8"))
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if not args.execute and not args.verify_only:
        print(
            json.dumps(
                {
                    "source_id": pins["source_id"],
                    "status": "PLAN_ONLY",
                    "files": len(pins["scoring_files"]),
                    "bytes": sum(item["bytes"] for item in pins["scoring_files"]),
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    original = sys.argv
    try:
        sys.argv = [original[0], "--output", str(args.output)]
        if args.resume:
            sys.argv.append("--resume")
        if args.verify_only:
            sys.argv.append("--verify-only")
        acquire(pins_path=PINS, source_id="NARRATIVEQA", parts=("scoring",))
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
