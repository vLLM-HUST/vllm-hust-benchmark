"""Verify or unpack the complete author QReCC collection outside the checkout.

Default is a read-only plan. This does not build an index or execute a task.
An interrupted extraction is retained; existing output is never overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = {
    "url": "https://zenodo.org/api/records/5760304/files/passages.zip/content",
    "bytes": 27512656777,
    "md5": "16c48fb9c4b9603b8ccc8ba3bd7d69c1",  # pragma: allowlist secret - public publisher checksum
    "entries": 56007,
    "files": 56003,
    "uncompressed_bytes": 86557816928,
    "maximum_member_bytes": 271773680,
}
DISK_RESERVE = 64 * 1024**3


def members(archive, spec):
    infos = archive.infolist()
    names = set()
    files = []
    for info in infos:
        name = PurePosixPath(info.filename)
        mode = stat.S_IFMT(info.external_attr >> 16)
        if (
            name.is_absolute()
            or info.orig_filename != info.filename
            or ".." in name.parts
            or "\\" in info.filename
            or ":" in info.filename
            or name.as_posix().rstrip("/") != info.filename.rstrip("/")
            or info.filename in names
            or info.flag_bits & 1
            or mode not in (0, stat.S_IFREG, stat.S_IFDIR)
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        ):
            raise ValueError("unsafe or duplicate original ZIP member")
        names.add(info.filename)
        if not info.is_dir():
            if name.suffix != ".jsonl":
                raise ValueError("original JSONL collection required")
            files.append(info)
    if (
        len(infos) != spec["entries"]
        or len(files) != spec["files"]
        or sum(x.file_size for x in files) != spec["uncompressed_bytes"]
        or max((x.file_size for x in files), default=0) != spec["maximum_member_bytes"]
    ):
        raise ValueError("complete original ZIP inventory required")
    return infos


def prepare(archive, *, output=None, max_bytes=0, spec=None):
    spec = SPEC if spec is None else spec
    archive = Path(archive)
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("original regular archive required")
    if output is not None:
        output = Path(output)
        if output.exists() or output.is_symlink():
            raise ValueError("fresh output required; retain previous failed extraction")
        output = output.resolve()
        if output.is_relative_to(ROOT):
            raise ValueError("raw collection must remain outside checkout")
        if max_bytes < spec["uncompressed_bytes"]:
            raise ValueError("explicit budget must cover the complete collection")
        ancestor = output.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < spec["uncompressed_bytes"] + DISK_RESERVE:
            raise ValueError("insufficient free space including shared disk reserve")
    with archive.open("rb") as source:
        before = os.fstat(source.fileno())
        if before.st_size != spec["bytes"]:
            raise ValueError("complete original archive byte count required")
        md5 = hashlib.md5(usedforsecurity=False)
        sha = hashlib.sha256()
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            md5.update(block)
            sha.update(block)
        if md5.hexdigest() != spec["md5"]:
            raise ValueError("publisher archive checksum mismatch")
        source.seek(0)
        with zipfile.ZipFile(source) as z:
            infos = members(z, spec)
            receipt = {
                "source_id": "QRECC",
                "status": "ORIGINAL_ARCHIVE_IDENTITY_AND_DIRECTORY_VERIFIED_NOT_MEMBER_CRC",
                "archive_bytes": before.st_size,
                "archive_sha256": sha.hexdigest(),
                "publisher_md5_verified": True,
                "entries": len(infos),
                "jsonl_files": spec["files"],
                "uncompressed_bytes": spec["uncompressed_bytes"],
                "member_crc_verified": False,
                "index_built": False,
                "native_tasks_executed": 0,
                "model_calls": 0,
                "source_checkbox_eligible": False,
                "collection_relative_path": "collection",
            }
            if output is not None:
                output.mkdir(parents=True, mode=0o700)
                marker = output / "collection-preparation.private.json"
                receipt.update(
                    status="EXTRACTING_COMPLETE_ORIGINAL_COLLECTION", completed_files=0
                )
                marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                try:
                    for info in infos:
                        dest = (output / "collection").joinpath(
                            *PurePosixPath(info.filename).parts
                        )
                        if info.is_dir():
                            dest.mkdir(parents=True, exist_ok=True)
                            continue
                        if (
                            shutil.disk_usage(output).free
                            < DISK_RESERVE + info.file_size
                        ):
                            raise ValueError("shared disk reserve would be exceeded")
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(info) as inp, dest.open("xb") as out:
                            shutil.copyfileobj(inp, out, length=4 * 1024**2)
                        if dest.stat().st_size != info.file_size:
                            raise ValueError("original member size differs")
                        receipt["completed_files"] += 1
                    current = os.fstat(source.fileno())
                    if (before.st_size, before.st_mtime_ns) != (
                        current.st_size,
                        current.st_mtime_ns,
                    ):
                        raise ValueError("archive changed during extraction")
                    receipt.update(
                        status="COMPLETE_ORIGINAL_COLLECTION_EXTRACTED_CRC_VERIFIED_NOT_INDEXED",
                        member_crc_verified=True,
                    )
                except Exception as error:
                    receipt.update(
                        status="FAILED_EXTRACTION_RETAINED_NOT_INDEXED",
                        error_type=type(error).__name__,
                    )
                    marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                    raise
                marker.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        after = os.fstat(source.fileno())
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("archive changed during preparation")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=("plan", "verify", "extract"), default="plan"
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-uncompressed-bytes", type=int, default=0)
    args = parser.parse_args()
    if args.action == "plan":
        print(
            json.dumps(
                {
                    "source_id": "QRECC",
                    "status": "PLAN_ONLY",
                    "archive": SPEC,
                    "shared_disk_reserve_bytes": DISK_RESERVE,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return
    if args.archive is None or (args.action == "extract" and args.output is None):
        parser.error("--archive required; extraction also requires --output")
    if args.action == "verify" and args.output is not None:
        parser.error("verification is read-only; do not pass --output")
    print(
        json.dumps(
            prepare(
                args.archive, output=args.output, max_bytes=args.max_uncompressed_bytes
            )
        )
    )


if __name__ == "__main__":
    main()
