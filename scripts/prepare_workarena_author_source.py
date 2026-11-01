"""Prepare fixed original WorkArena software, never obtain instance credentials."""

import argparse
import json
from pathlib import Path

from prepare_jsonschemabench_author_source import verify_source
from prepare_swebench_author_source import prepare

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/workarena-author-source.json"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    args = parser.parse_args()
    pins = json.loads(PINS.read_text("utf-8"))
    if args.action == "prepare":
        if args.archive is None:
            parser.error("existing fixed official source archive required")
        prepare(args.archive, args.author_root, pins)
    if args.action != "plan":
        verify_source(args.author_root, PINS)
    print(
        json.dumps(
            {
                "source_id": "WORKARENA",
                "revision": pins["revision"],
                "official_archive_url": f"https://codeload.github.com/{pins['repository']}/tar.gz/{pins['revision']}",
                "files": len(pins["files"]),
                "bytes": sum(row["bytes"] for row in pins["files"]),
                "status": "PLAN_ONLY"
                if args.action == "plan"
                else "ORIGINAL_PUBLIC_ASSETS_VERIFIED_NOT_EXECUTED",
                "instance_access": "Requires separately authorized instance pool; this command neither accepts terms nor retrieves credentials",
                "model_calls": 0,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
