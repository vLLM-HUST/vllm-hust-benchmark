"""Acquire or verify original MS MARCO passage reranking assets, never run a model."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/msmarco-original-reranking-assets.json"
)


def verify_file(path, expected):
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != expected["bytes"]
    ):
        raise ValueError(f"original asset type or size differs: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    if digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"original snapshot checksum differs: {path.name}")


def acquire_file(root, expected):
    name = expected["name"]
    if Path(name).name != name or name not in {"qrels.dev.tsv", "top1000.dev.tar.gz"}:
        raise ValueError("unsupported original filename")
    parsed = urlsplit(expected["url"])
    if parsed.scheme != "https" or parsed.netloc != "msmarco.z22.web.core.windows.net":
        raise ValueError("original publisher HTTPS endpoint required")
    target = root / name
    if target.exists():
        verify_file(target, expected)
        return
    temporary = root / (name + ".part")
    # Failed partial downloads remain for diagnosis; never overwrite or silently resume.
    with (
        temporary.open("xb") as output,
        urlopen(expected["url"], timeout=90) as response,
    ):
        if response.status != 200 or response.geturl() != expected["url"]:
            raise ValueError("unexpected original download response or redirect")
        size = 0
        while chunk := response.read(1024**2):
            size += len(chunk)
            if size > expected["bytes"]:
                raise ValueError("original download exceeds pinned size")
            output.write(chunk)
    verify_file(temporary, expected)
    if target.exists():
        raise FileExistsError("original target appeared during acquisition")
    # Atomic no-clobber publication on the same filesystem; retain partial on failure.
    os.link(temporary, target)
    temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--accept-noncommercial-research-license", action="store_true")
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text("utf-8"))
    if not (args.download or args.verify):
        print(
            json.dumps({"status": "OFFLINE_PLAN_NO_DATA_ACQUIRED", **profile}, indent=2)
        )
        return
    if args.data_dir is None:
        parser.error("--data-dir is required for download or verification")
    if args.download and not args.accept_noncommercial_research_license:
        parser.error("review original terms and explicitly accept the research license")
    if args.download:
        args.data_dir.mkdir(parents=True, exist_ok=True)
    for item in profile["files"]:
        if args.download:
            acquire_file(args.data_dir, item)
        else:
            verify_file(args.data_dir / item["name"], item)
    print(
        json.dumps(
            {
                "source_id": "MSMARCO_PASSAGE",
                "status": "ORIGINAL_ASSETS_VERIFIED_NATIVE_EXECUTION_PENDING",
                "files": len(profile["files"]),
                "model_calls": 0,
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
