"""Run pinned author download/preparation/validation, never memory inference.

Original questions, trajectories, screenshots and haystacks stay external.
Download uses the unchanged publisher script and an explicit fixed revision.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

from prepare_jsonschemabench_author_source import verify_source

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PINS = ROOT / "configs/workload-native/longmemeval-v2-author-source.json"
DATA_PINS = ROOT / "configs/workload-native/longmemeval-v2-original-assets.json"


def verify_assets(root, pins):
    root = Path(root).resolve(strict=True)
    for item in pins["files"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe pinned asset path")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("original file must remain within external data root")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"original asset size differs: {relative}")
        if "sha256" in item:
            digest = hashlib.sha256()
        else:
            digest = hashlib.sha1(f"blob {item['bytes']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item.get("sha256", item.get("git_blob")):
            raise ValueError(f"original asset identity differs: {relative}")


def archive_capacity(root):
    total = 0
    for name in (
        "trajectory_screenshots/web_screenshots.tar.gz",
        "trajectory_screenshots/enterprise_screenshots_base.tar.gz",
    ):
        with tarfile.open(root / name, "r:gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError("unsafe original archive member; no extraction")
                total += member.size
    return total


def run(args):
    pins = json.loads(DATA_PINS.read_text("utf-8"))
    total = sum(item["bytes"] for item in pins["files"])
    receipt = {
        "source_id": "LONGMEMEVAL_V2",
        "status": "PLAN_ONLY_NO_EXECUTION",
        "revision": pins["revision"],
        "files": len(pins["files"]),
        "bytes": total,
        "model_calls": 0,
        "native_benchmark_verified": False,
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
        "commands": [],
    }
    if args.action == "plan":
        return receipt
    source = verify_source(args.author_root, SOURCE_PINS)
    data = args.data_root.resolve()
    output = args.output.resolve()
    if any(
        left.is_relative_to(right)
        for left, right in (
            (data, ROOT),
            (output, ROOT),
            (source, data),
            (data, source),
            (source, output),
            (output, source),
            (output, data),
            (data, output),
        )
    ):
        raise ValueError(
            "external original data, author source and fresh output must be separate"
        )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)

    def save():
        (output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    def original(command):
        position = len(receipt["commands"])
        receipt["commands"].append({"author_script": command[1], "exit_code": None})
        save()
        with (output / f"step-{position}.private.log").open("xb") as log:
            result = subprocess.run(
                command, cwd=source, stdout=log, stderr=subprocess.STDOUT
            )
        receipt["commands"][-1]["exit_code"] = result.returncode
        save()
        if result.returncode:
            raise RuntimeError(
                "original author command failed; preserve log, do not replace data"
            )

    try:
        if args.action == "download":
            if args.max_bytes is None or args.max_bytes < total:
                raise ValueError(
                    f"explicit original-byte budget at least {total} required"
                )
            parent = data.parent
            if not parent.is_dir() or shutil.disk_usage(parent).free < total * 3:
                raise ValueError(
                    "existing external parent with 3x compressed-byte free capacity required"
                )
            if data.exists():
                raise ValueError(
                    "fresh data root required; verify existing assets, never force-delete"
                )
            original(
                [
                    sys.executable,
                    str(source / "data/download_data.py"),
                    "--repo-id",
                    pins["repository"],
                    "--revision",
                    pins["revision"],
                    "--data-root",
                    str(data),
                ]
            )
        verify_assets(data, pins)
        receipt["original_file_identities_verified"] = True
        if args.action == "prepare":
            uncompressed = archive_capacity(data)
            receipt["archive_uncompressed_bytes"] = uncompressed
            if shutil.disk_usage(data).free < uncompressed + 1024**3:
                raise ValueError(
                    "insufficient free space for original screenshots plus 1GiB reserve"
                )
            original(
                [
                    sys.executable,
                    str(source / "data/prepare_data.py"),
                    "--data-root",
                    str(data),
                    "--mode",
                    "symlink",
                ]
            )
            for tier in ("small", "medium"):
                original(
                    [
                        sys.executable,
                        str(source / "data/validate_data.py"),
                        "--data-root",
                        str(data),
                        "--tier",
                        tier,
                    ]
                )
            receipt["both_original_tiers_validated"] = True
        verify_source(source, SOURCE_PINS)
        receipt["status"] = "ORIGINAL_DATA_PREPARATION_VERIFIED_NOT_NATIVE_BENCHMARK"
    except Exception as exc:
        receipt.update(
            status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=["plan", "download", "verify", "prepare"], default="plan"
    )
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
