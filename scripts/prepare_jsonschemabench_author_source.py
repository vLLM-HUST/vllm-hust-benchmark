"""Acquire or verify original author files without requiring Git transport.

Pinned Git blob identities, sizes and SHA256 are required for every file.
No author code patching, fake modules, dataset downloads or model calls.
"""

import argparse
import json
from pathlib import Path

from prepare_crosscodeeval_original_assets import acquire, verify_git_blob
from prepare_xsum_original_assets import bound_path

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/jsonschemabench-author-source.json"
)


def verify_source(root, pins_path=None):
    pins = json.loads((pins_path or PINS).read_text(encoding="utf-8"))
    root = Path(root).resolve(strict=True)
    for item in pins["files"]:
        verify_git_blob(bound_path(root, "", item), item)
    expected_python = {
        item["path"] for item in pins["files"] if item["path"].endswith(".py")
    }
    actual_python = {path.relative_to(root).as_posix() for path in root.rglob("*.py")}
    if actual_python != expected_python:
        raise ValueError(
            "unexpected/missing author Python files, no source substitution"
        )
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=["jsonschemabench", "bigcodebench", "longmemeval-v2"],
        default="jsonschemabench",
        help="Pinned original author source only; historical default unchanged",
    )
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()
    pins_path = PINS.with_name(f"{args.source}-author-source.json")
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    total = sum(item["bytes"] for item in pins["files"])
    if args.action == "prepare":
        if args.max_bytes is None or args.max_bytes < total:
            parser.error(f"explicit source budget at least {total} bytes required")
        for original in pins["files"]:
            item = {
                **original,
                "url": f"https://api.github.com/repos/{pins['repository']}/git/blobs/{original['git_blob']}",
            }
            target = bound_path(args.author_root, "", item)
            if target.exists():
                verify_git_blob(target, item)
            else:
                acquire(target, item)
    if args.action != "plan":
        verify_source(args.author_root, pins_path)
    print(
        json.dumps(
            {
                "source_id": pins["source_id"],
                "status": "PLAN_ONLY_NO_DOWNLOAD"
                if args.action == "plan"
                else "ORIGINAL_PUBLISHER_SOURCE_FILES_VERIFIED_NOT_NATIVE_EXECUTION",
                "revision": pins["revision"],
                "files": len(pins["files"]),
                "bytes": total,
                "model_calls": 0,
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
