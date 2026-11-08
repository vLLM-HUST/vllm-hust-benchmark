"""Acquire/check original full OGB archive; never extract or execute a task."""

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_mlagentbench_cifar_assets import acquire, verify

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/mlagentbench-ogbn-arxiv-original-assets.json"
)


def prepare(args):
    report = {
        "source_id": "MLAGENTBENCH",
        "task": "ogbn-arxiv",
        "status": "PLAN_ONLY_NO_DOWNLOAD_OR_EXECUTION",
        "native_tasks_run": 0,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if args.action == "plan":
        return report
    profile = json.loads(PROFILE.read_text("utf-8"))
    spec = profile["data"]
    if args.action == "acquire":
        acquire(args.data_root, spec, max_bytes=args.max_bytes, resume=args.resume)
    archive = args.data_root / spec["path"]
    asset = verify(archive, spec)
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if (
            len(names) != profile["zip_members"]
            or len(set(names)) != len(names)
            or any(
                not name.startswith("arxiv/") or ".." in PurePosixPath(name).parts
                for name in names
            )
            or sum(item.file_size for item in source.infolist())
            != profile["zip_uncompressed_bytes"]
            or source.testzip() is not None
        ):
            raise ValueError("pinned full original OGB archive/CRC contract differs")
    report.update(
        status="FULL_ORIGINAL_OGB_ARCHIVE_VERIFIED_NOT_EXTRACTED_OR_EXECUTED",
        asset=asset,
        digest_origin=profile["digest_origin"],
        zip_members=len(names),
        all_original_zip_crc_passed=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "acquire", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int, default=128 * 1024**2)
    parser.add_argument("--resume", action="store_true")
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
