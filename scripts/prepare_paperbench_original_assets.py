"""Prepare unchanged published PaperBench paper assets externally; never run tasks."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.request import Request, urlopen

from prepare_crosscodeeval_original_assets import (
    acquire,
    git_lfs_pointer,
    verify_git_blob,
)
from prepare_xsum_original_assets import bound_path

ROOT = Path(__file__).resolve().parents[1]
PINS = (
    ROOT / "workload-preparation/data/catalogs/paperbench_original_paper_assets_v1.json"
)
REVISION = "b78dab1da774a6456b1ed9c48c9ff194ee3ff36d"  # pragma: allowlist secret - fixed official public revision
PREFIX = "project/paperbench/data/papers/"


def load_pins(path=PINS):
    pins = json.loads(Path(path).read_text("utf-8"))
    if (
        pins.get("source_id") != "PAPERBENCH"
        or pins.get("repository") != "openai/frontier-evals"
        or pins.get("revision") != REVISION
    ):
        raise ValueError("fixed official PaperBench identity required")
    seen = set()
    for item in pins["files"]:
        name = item["path"]
        relative = PurePosixPath(name)
        if (
            not name.startswith(PREFIX)
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or ":" in name
            or name != relative.as_posix()
            or name in seen
            or type(item["bytes"]) is not int
            or not 0 <= item["bytes"] <= 64 * 1024**2
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            or re.fullmatch(r"[0-9a-f]{40}", item["git_blob"]) is None
        ):
            raise ValueError("distinct contained original asset identities required")
        seen.add(name)
        pointer = git_lfs_pointer(item)
        if pointer is not None:
            blob = hashlib.sha1(b"blob " + str(len(pointer)).encode() + b"\0" + pointer)
            if blob.hexdigest() != item["git_blob"]:
                raise ValueError("original LFS pointer Git identity differs")
    if len(seen) != 506 or sum(x["bytes"] for x in pins["files"]) != 213434104:
        raise ValueError("complete fixed published paper asset selection required")
    return pins


def fetch_original_payload(path, item, url):
    """Bound response bytes before publishing; retain failed partials for inspection."""
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise ValueError("inspect existing final/partial; never overwrite")
    with urlopen(
        Request(url, headers={"User-Agent": "original-PaperBench-assets"}), timeout=90
    ) as response:
        path.parent.mkdir(parents=True, exist_ok=True)
        remaining = item["bytes"]
        with partial.open("xb") as stream:
            while True:
                block = response.read(min(1024**2, remaining + 1))
                if not block:
                    break
                if len(block) > remaining:
                    raise ValueError("publisher payload exceeds pinned size budget")
                stream.write(block)
                remaining -= len(block)
    verify_git_blob(partial, item)
    os.link(partial, path)
    partial.unlink()


def prepare(output, pins, *, github_api=False):
    for item in pins["files"]:
        path = bound_path(output, "", item)
        if path.exists():
            verify_git_blob(path, item)
            continue
        if github_api and "git_lfs" not in item:
            acquire(
                path,
                {
                    **item,
                    "url": f"https://api.github.com/repos/{pins['repository']}/git/blobs/{item['git_blob']}",
                },
            )
        else:
            endpoint = (
                "https://media.githubusercontent.com/media"
                if "git_lfs" in item
                else "https://raw.githubusercontent.com"
            )
            fetch_original_payload(
                path,
                item,
                f"{endpoint}/{pins['repository']}/{pins['revision']}/{item['path']}",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument(
        "--github-api",
        action="store_true",
        help="Official blob API for non-LFS files only; respect its access/rate limits",
    )
    args = parser.parse_args()
    pins = load_pins()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("raw assets must remain outside the checkout")
    total = sum(x["bytes"] for x in pins["files"])
    if args.action == "prepare":
        if args.max_bytes is None or args.max_bytes < total:
            raise ValueError("explicit byte budget must cover all pinned paper assets")
        prepare(output, pins, github_api=args.github_api)
    if args.action != "plan":
        for item in pins["files"]:
            verify_git_blob(bound_path(output, "", item), item)
    print(
        json.dumps(
            {
                "source_id": "PAPERBENCH",
                "revision": pins["revision"],
                "files": len(pins["files"]),
                "bytes": total,
                "status": "PLAN_ONLY"
                if args.action == "plan"
                else "ORIGINAL_PUBLISHED_ASSETS_VERIFIED_NOT_NATIVE_EXECUTION",
                "paperbench_data_dir": str(output / "project/paperbench/data"),
                "external_scientific_datasets_qualified": False,
                "model_calls": 0,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
