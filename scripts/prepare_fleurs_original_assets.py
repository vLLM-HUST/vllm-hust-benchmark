"""Plan or acquire pinned original FLEURS Parquet files, outside the repository."""

import argparse
import json
import os
from pathlib import Path

from prepare_xsum_original_assets import bound_path, sha, verify
from urllib.request import Request, urlopen

from vllm_hust_workload_data.registry import source_spec


def select_files(source, split, locales):
    known = {item["locale"] for item in source["files"]}
    if locales and not set(locales) <= known:
        raise ValueError("unknown locale; aggregate all is not a separate locale")
    return [
        item
        for item in source["files"]
        if item["split"] == split and (not locales or item["locale"] in locales)
    ]


def acquire(path, item, endpoint):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    # Never overwrite an active or abandoned partial from another invocation.
    if partial.exists():
        raise ValueError(
            "existing partial requires ownership/terminal-state inspection"
        )
    url = item["url"].replace("https://huggingface.co", endpoint)
    with urlopen(
        Request(url, headers={"User-Agent": "FLEURS-original-assets"}), timeout=60
    ) as response:
        with partial.open("xb") as output:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                output.write(block)
    verify(partial, item)
    if path.exists():
        raise ValueError("destination appeared during acquisition; do not overwrite")
    # Atomic no-clobber publication on the same filesystem, including Linux.
    os.link(partial, path)
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="test"
    )
    parser.add_argument("--locales", nargs="+")
    parser.add_argument(
        "--include-model",
        action="store_true",
        help="also acquire the fixed multilingual Whisper checkpoint, without loading it",
    )
    parser.add_argument(
        "--action", choices=["plan", "download", "verify"], default="plan"
    )
    parser.add_argument(
        "--max-bytes", type=int, help="explicit total selected download budget"
    )
    parser.add_argument(
        "--hf-endpoint",
        choices=["https://huggingface.co", "https://hf-mirror.com"],
        default="https://huggingface.co",
    )
    args = parser.parse_args()
    source = source_spec("FLEURS")
    selected = select_files(source, args.split, args.locales)
    if args.include_model:
        profile = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs/workload-native/fleurs-whisper-profile.json"
            ).read_text(encoding="utf-8")
        )
        selected.append(
            {
                "path": "models/" + profile["checkpoint"],
                "bytes": profile["checkpoint_bytes"],
                "sha256": profile["checkpoint_sha256"],
                "url": profile["checkpoint_url"],
            }
        )
    total = sum(item["bytes"] for item in selected)
    if args.action == "download" and (args.max_bytes is None or args.max_bytes < total):
        raise ValueError("explicit max-bytes must cover all selected original assets")
    receipt = {
        "source_id": "FLEURS",
        "revision": source["revision"],
        "action": args.action,
        "split": args.split,
        "selected_files": len(selected),
        "selected_bytes": total,
        "verified": [],
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if args.action != "plan":
        for item in selected:
            path = bound_path(args.output, "", item)
            if args.action == "download" and not path.exists():
                acquire(path, item, args.hf_endpoint)
            verify(path, item)
            receipt["verified"].append({"path": item["path"], "sha256": sha(path)})
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
