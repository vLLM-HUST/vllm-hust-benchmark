"""Explicit pinned publisher archive preparation; no invented prompts or requests."""

import argparse
import json
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-archive", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    from vllm_hust_workload_data.registry import source_spec
    from vllm_hust_workload_data.assets import verify_file

    source = source_spec("SCHEMASTORE")
    (asset,) = source["files"]
    if not args.execute and not args.verify_only:
        print(
            json.dumps(
                {
                    "source_id": "SCHEMASTORE",
                    "status": "PLAN_ONLY",
                    "revision": source["revision"],
                    "bytes": asset["bytes"],
                    "url": asset["url"],
                    "model_calls": 0,
                }
            )
        )
        return
    if args.verify_only:
        verify_file(args.output / "data", asset)
        print(
            json.dumps(
                {"status": "ORIGINAL_ARCHIVE_VERIFIED_READ_ONLY", "model_calls": 0}
            )
        )
        return
    if args.output.exists():
        raise ValueError("fresh external preparation output required")
    root = args.output / "data"
    root.mkdir(mode=0o700, parents=True)
    target = root / asset["path"]
    temporary = target.with_name(target.name + ".partial")
    if args.reuse_archive:
        verify_file(
            args.reuse_archive.parent, {**asset, "path": args.reuse_archive.name}
        )
        shutil.copyfile(args.reuse_archive, temporary)
    else:
        with (
            urlopen(
                Request(
                    asset["url"],
                    headers={"User-Agent": "original-schema-asset-preparation"},
                ),
                timeout=60,
            ) as response,
            temporary.open("xb") as stream,
        ):
            total = 0
            while block := response.read(1024**2):
                total += len(block)
                if total > asset["bytes"]:
                    raise ValueError("publisher archive exceeds pinned size")
                stream.write(block)
    verify_file(root, {**asset, "path": temporary.name})
    temporary.rename(target)
    result = {
        "source_id": "SCHEMASTORE",
        "status": "PINNED_ORIGINAL_ARCHIVE_PREPARED_NOT_NATIVE_VALIDATION",
        "original_bytes": asset["bytes"],
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    (args.output / "receipt-public.json").write_text(
        json.dumps(result, indent=2), "utf-8"
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
