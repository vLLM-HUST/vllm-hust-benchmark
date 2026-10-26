"""Explicit original PAWS HF distribution acquisition, not benchmark execution."""

import argparse
import json
from pathlib import Path
import sys

from prepare_xsum_original_assets import main as acquire

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/paws-original-assets.json"
)


def plan():
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    files = pins["dataset"]["files"]
    if pins["source_id"] != "PAWS" or len({item["path"] for item in files}) != len(
        files
    ):
        raise ValueError("Distinct original PAWS asset identities required")
    return {
        "source_id": "PAWS",
        "status": "PLAN_ONLY_NO_DOWNLOAD_OR_NATIVE_EXECUTION",
        "revision": pins["dataset"]["revision"],
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "configurations": sorted({item["path"].split("/")[0] for item in files}),
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--hf-endpoint",
        choices=["https://huggingface.co", "https://hf-mirror.com"],
        default="https://huggingface.co",
    )
    args = parser.parse_args()
    summary = plan()
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if not args.execute and not args.verify_only:
        print(json.dumps(summary, indent=2))
        return
    arguments = [
        sys.argv[0],
        "--output",
        str(args.output),
        "--hf-endpoint",
        args.hf_endpoint,
    ]
    if args.resume:
        arguments.append("--resume")
    if args.verify_only:
        arguments.append("--verify-only")
    original = sys.argv
    try:
        sys.argv = arguments
        acquire(
            pins_path=PINS, source_id="PAWS", parts=("data",), hf_download_query=True
        )
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
