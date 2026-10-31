"""Prepare one pinned original ZIP member, never a whole archive or model run."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
from urllib.parse import quote
from urllib.request import urlopen
import zlib

from prepare_wavcaps_original_media import fetch
from vllm_hust_workload_data.assets import verify_file

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/streamingbench-author-source.json"
)
CHUNK = 8 * 1024**2


def transfer(member, archive, inventory, sink, *, opener=urlopen):
    if member["method"] != 8 or member["flags"] & ~0x808:
        raise ValueError("only original unencrypted deflated member supported")

    def read_range(start, end):
        if not 0 <= start <= end < archive["bytes"] or end - start + 1 > CHUNK:
            raise ValueError("bounded original ZIP range required")

        def checked_open(request, *, timeout):
            response = opener(request, timeout=timeout)
            if response.headers.get("ETag") != archive["captured_etag"]:
                response.close()
                raise ValueError("original archive differs from captured ETag")
            return response

        return fetch(
            {
                "url": f"https://huggingface.co/datasets/{inventory['dataset_repo']}/resolve/{inventory['dataset_revision']}/{quote(archive['archive'])}?member_range={start}",
                "start": start,
                "end": end,
                "part_bytes": archive["bytes"],
            },
            opener=checked_open,
        )

    offset = member["local_header_offset"]
    raw = read_range(offset, offset + 511)
    header = struct.unpack_from("<4s5H3I2H", raw)
    if (
        header[0] != b"PK\x03\x04"
        or header[2] != member["flags"]
        or header[3] != member["method"]
    ):
        raise ValueError("original local header differs from central directory")
    name_size, extra_size = header[9:11]
    if 30 + name_size + extra_size > len(raw):
        raise ValueError("original local header exceeds bounded preflight")
    name = raw[30 : 30 + name_size].decode("utf-8" if header[2] & 2048 else "cp437")
    if name != member["archive_member"]:
        raise ValueError("original ZIP member name differs")
    # Bit3 stores CRC/sizes in the data descriptor, outside compressed payload.
    if not header[2] & 8 and tuple(header[6:9]) != (
        member["crc32"],
        member["compressed_bytes"],
        member["bytes"],
    ):
        raise ValueError("original local CRC or sizes differ")
    start = offset + 30 + name_size + extra_size
    end = start + member["compressed_bytes"] - 1
    decoder, digest, count, crc = zlib.decompressobj(-15), hashlib.sha256(), 0, 0
    for position in range(start, end + 1, CHUNK):
        compressed = read_range(position, min(position + CHUNK - 1, end))
        while compressed:
            decoded = decoder.decompress(
                compressed, min(CHUNK, member["bytes"] - count + 1)
            )
            count += len(decoded)
            if count > member["bytes"] or decoder.unused_data:
                raise ValueError("original member exceeds declared size or stream")
            if sink.write(decoded) != len(decoded):
                raise OSError("incomplete original video write")
            digest.update(decoded)
            crc = zlib.crc32(decoded, crc)
            compressed = decoder.unconsumed_tail
    if not decoder.eof or count != member["bytes"] or crc != member["crc32"]:
        raise ValueError("original member full size, deflate EOF or CRC differs")
    return {
        "bytes": count,
        "observed_sha256": digest.hexdigest(),
        "crc32": crc,
        "original_complete_member_verified": True,
    }


def verify_existing(member, inventory, media_root):
    name = PurePosixPath(member["original_task_media_path"]).name
    qualified = [
        item
        for item in inventory.get("qualified_original_members", [])
        if item["path"] == name
    ]
    if len(qualified) != 1 or qualified[0]["bytes"] != member["bytes"]:
        raise ValueError("existing original has no qualified SHA256; never overwrite")
    report = verify_file(media_root, qualified[0])
    return {
        **report,
        "file": name,
        "existing_original_reused": True,
        "network_fetch_executed": False,
    }


def download(member, archive, inventory, media_root, *, max_bytes):
    root = Path(media_root).resolve()
    if root.is_relative_to(PROFILE.parents[2]) or max_bytes < member["bytes"]:
        raise ValueError(
            "external root and explicit complete-member byte budget required"
        )
    name = PurePosixPath(member["original_task_media_path"]).name
    if not name or Path(name).name != name or "\\" in name or ":" in name:
        raise ValueError("original media basename required")
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise ValueError("inspect existing original/partial; never overwrite")
    if target.exists():
        return verify_existing(member, inventory, root)
    if shutil.disk_usage(root).free < member["bytes"] + 512 * 1024**2:
        raise ValueError("insufficient space including 512MiB reserve")
    with partial.open("xb") as stream:
        report = transfer(member, archive, inventory, stream)
    verify_file(
        root,
        {
            "path": partial.name,
            "bytes": report["bytes"],
            "sha256": report["observed_sha256"],
        },
    )
    # Atomic no-overwrite publication, retaining the failed partial on errors.
    os.link(partial, target)
    partial.unlink()
    return {**report, "file": name}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member", required=True, help="Exact registered native video basename"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Read-only qualified original SHA256 verification; no network",
    )
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--max-bytes", type=int, default=0)
    args = parser.parse_args()
    inventory = json.loads(PROFILE.read_text("utf-8"))["media_archive_inventory"]
    matches = [
        item
        for item in inventory["selected_original_video_members"]
        if PurePosixPath(item["original_task_media_path"]).name == args.member
    ]
    if len(matches) != 1:
        parser.error("one exact registered original native member required")
    member = matches[0]
    archive = next(
        item for item in inventory["archives"] if item["archive"] == member["archive"]
    )
    report = {"member": member, "archive_bytes": archive["bytes"]}
    if args.verify:
        if args.media_root is None:
            parser.error("explicit existing --media-root required")
        report = verify_existing(member, inventory, args.media_root)
    if args.download:
        if args.media_root is None:
            parser.error("explicit external --media-root required")
        report = download(
            member, archive, inventory, args.media_root, max_bytes=args.max_bytes
        )
    print(
        json.dumps(
            {
                "source_id": "STREAMINGBENCH",
                "status": "ORIGINAL_MEDIA_VERIFIED_READ_ONLY_NOT_NATIVE_EXECUTION"
                if args.verify
                else "ORIGINAL_MEDIA_PREPARED_NOT_NATIVE_EXECUTION"
                if args.download
                else "PLAN_ONLY_NO_NETWORK",
                **report,
                "model_calls": 0,
                "native_tasks_run": 0,
                "source_checkbox_eligible": False,
            }
        )
    )
