"""Check captured original TAR-header relations and plan unchanged clip ranges.

This does not authenticate a fabricated index against the publisher, download
media, rehash complete parts, decode video, or run a benchmark. Capture identity
and HTTP provenance must be retained separately from structural validation.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from prepare_ovobench_native_selection import prepare

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/ovobench-author-source.json"
)


def part_ranges(start, length, parts):
    """Map complete original member bytes across the publisher split boundaries."""
    if type(start) is not int or type(length) is not int or start < 0 or length <= 0:
        raise ValueError("positive original member byte span required")
    result, offset, remaining = [], 0, length
    for part in parts:
        if offset <= start < offset + part["bytes"]:
            local = start - offset
            count = min(remaining, part["bytes"] - local)
            result.append(
                {"part": part["path"], "start": local, "end": local + count - 1}
            )
            remaining -= count
            start += count
        offset += part["bytes"]
        if not remaining:
            return result
    raise ValueError("original member exceeds registered publisher parts")


def plan(index, selection, inventory):
    parts = [
        part
        for part in inventory["parts"]
        if part["path"].startswith("chunked_videos.")
    ]
    entries = index.get("entries")
    if not isinstance(entries, list) or len(entries) > 4096:
        raise ValueError("bounded captured TAR header entries required")
    found, names, position = {}, set(), 0
    expected = {
        "chunked_videos/" + item["path"]: item
        for item in selection["required_native_clips"]
    }
    if len(expected) != selection["required_native_clip_count"]:
        raise ValueError("distinct original selected clips required")
    for entry in entries:
        raw = base64.b64decode(entry["header_base64"], validate=True)
        if len(raw) != 512 or hashlib.sha256(raw).hexdigest() != entry["header_sha256"]:
            raise ValueError("captured original header identity mismatch")
        item = tarfile.TarInfo.frombuf(raw, encoding="utf-8", errors="strict")
        name = PurePosixPath(item.name)
        if (
            not (item.isdir() or item.isfile())
            or item.size < 0
            or name.is_absolute()
            or ".." in name.parts
            or item.name in names
            or item.name != entry["path"]
            or item.size != entry["bytes"]
            or entry["directory"] != item.isdir()
            or entry["header_offset"] != position
            or entry["data_offset"] != position + 512
        ):
            raise ValueError("captured TAR member chain or identity differs")
        header = part_ranges(position, 512, parts)
        if (
            len(header) != 1
            or header[0]["part"] != entry["part"]
            or not index.get("etags", {}).get(entry["part"])
        ):
            raise ValueError("captured header part or observed ETag missing")
        names.add(item.name)
        if item.name in expected:
            if not item.isfile() or item.size <= 0:
                raise ValueError(
                    "selected original media must be a nonempty regular member"
                )
            found[item.name] = {
                **expected[item.name],
                "archive_member": item.name,
                "bytes": item.size,
                "ranges": part_ranges(position + 512, item.size, parts),
            }
        position += 512 + ((item.size + 511) // 512) * 512
        if position > sum(part["bytes"] for part in parts):
            raise ValueError("captured TAR member exceeds original archive bounds")
    complete = index.get("complete") is True
    zeros = index.get("zero_headers")
    if (
        type(zeros) is not int
        or zeros not in ({2} if complete else {0, 1})
        or index.get("next_offset") != position + 512 * zeros
    ):
        raise ValueError("captured TAR checkpoint/end-marker relation differs")
    missing = [name for name in expected if name not in found]
    return {
        "source_id": "OVOBENCH",
        "status": "CAPTURED_HEADER_STRUCTURE_AND_ORIGINAL_RANGE_PLAN_ONLY",
        "captured_archive_index_complete": complete,
        "captured_header_entries": len(entries),
        "required_original_clips": len(expected),
        "located_original_clips": len(found),
        "missing_original_members": missing,
        "located_original_bytes": sum(v["bytes"] for v in found.values()),
        "members": [found[name] for name in expected if name in found],
        "fixed_selection_fully_located": not missing,
        "whole_part_publisher_hash_recomputed": False,
        "media_acquired": False,
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    with args.index.open("rb") as stream:
        raw = stream.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2:
        parser.error("captured header index exceeds byte budget")
    inventory = json.loads(PROFILE.read_text("utf-8"))["media_archive_inventory"]
    report = plan(json.loads(raw), prepare(args.data_root), inventory)
    if args.require_complete and not (
        report["captured_archive_index_complete"]
        and report["fixed_selection_fully_located"]
    ):
        parser.error(
            "original index/selection incomplete; no replacement clips allowed"
        )
    print(json.dumps(report))
