"""Plan or acquire pinned original MMMU assets outside the repository.

This is data acquisition, never model execution, scoring or source completion.
"""

import argparse
import json
from pathlib import Path
import sys

from prepare_xsum_original_assets import main as acquire

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/mmmu-original-assets.json"
)


def plan(pins_path=PINS):
    pins = json.loads(Path(pins_path).read_text(encoding="utf-8"))
    dataset = pins["dataset"]
    files = dataset["files"]
    paths = [item["path"] for item in files]
    if pins["source_id"] != "MMMU" or len(paths) != len(set(paths)):
        raise ValueError("Unique original MMMU assets required")
    for item in files:
        path = Path(item["path"])
        if path.is_absolute() or ".." in path.parts or "\\" in item["path"]:
            raise ValueError("Contained original relative paths required")
        if item["bytes"] <= 0 or len(item["sha256"]) != 64:
            raise ValueError("Original size and SHA256 required")
        int(item["sha256"], 16)
    return {
        "source_id": "MMMU",
        "status": "PLAN_ONLY_NO_DOWNLOAD_OR_NATIVE_EXECUTION",
        "repository": dataset["repo_id"],
        "revision": dataset["revision"],
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "subjects": len({path.split("/")[0] for path in paths}),
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
    if args.verify_only:
        arguments.append("--verify-only")
    if args.resume:
        arguments.append("--resume")
    previous = sys.argv
    try:
        sys.argv = arguments
        acquire(
            pins_path=PINS, source_id="MMMU", parts=("data",), hf_download_query=True
        )
    finally:
        sys.argv = previous


if __name__ == "__main__":
    main()
