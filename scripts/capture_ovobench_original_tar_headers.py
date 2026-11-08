"""Capture original publisher TAR headers, skipping all member payloads.

Default is a plan. Explicit capture writes a fresh external checkpoint directory.
Resume by supplying a previous --index and another fresh --output directory.
No media download, model call, whole-part hash claim or source completion.
"""

import argparse
import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from urllib.request import urlopen

from plan_ovobench_original_media_ranges import PROFILE, part_ranges, plan
from prepare_ovobench_native_selection import prepare
from prepare_wavcaps_original_media import fetch


def empty_index():
    return {
        "entries": [],
        "next_offset": 0,
        "etags": {},
        "complete": False,
        "zero_headers": 0,
    }


def advance(index, inventory, *, opener=urlopen):
    """One bounded read; failure never changes the supplied valid checkpoint."""
    if index["complete"]:
        return deepcopy(index)
    offset = index["next_offset"]
    if type(offset) is not int or offset < 0 or offset % 512:
        raise ValueError("aligned original TAR header offset required")
    parts = [p for p in inventory["parts"] if p["path"].startswith("chunked_videos.")]
    spans = part_ranges(offset, 512, parts)
    if len(spans) != 1:
        raise ValueError("original header must fit registered aligned part")
    span = spans[0]
    part = next(p for p in parts if p["path"] == span["part"])
    observed = {}

    def header_open(request, *, timeout):
        response = opener(request, timeout=timeout)
        etag = response.headers.get("ETag")
        if not etag or index["etags"].get(part["path"], etag) != etag:
            response.close()
            raise ValueError("publisher header part differs from captured ETag")
        observed["etag"] = etag
        return response

    raw = fetch(
        {
            "url": f"https://huggingface.co/datasets/{inventory['dataset_repo']}/resolve/{inventory['dataset_revision']}/{part['path']}?header_offset={span['start']}",
            "start": span["start"],
            "end": span["end"],
            "part_bytes": part["bytes"],
        },
        opener=header_open,
    )
    result = deepcopy(index)
    result.update(
        source_id="OVOBENCH",
        dataset_repo=inventory["dataset_repo"],
        dataset_revision=inventory["dataset_revision"],
    )
    result["etags"][part["path"]] = observed["etag"]
    if raw == bytes(512):
        result["zero_headers"] += 1
        result["next_offset"] += 512
        result["complete"] = result["zero_headers"] == 2
        result.setdefault("captured_end_markers", []).append(
            {
                "offset": offset,
                "part": part["path"],
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    else:
        if result["zero_headers"] or len(result["entries"]) >= 4096:
            raise ValueError(
                "unexpected member after end marker or header budget exceeded"
            )
        item = tarfile.TarInfo.frombuf(raw, encoding="utf-8", errors="strict")
        name = PurePosixPath(item.name)
        if (
            not (item.isdir() or item.isfile())
            or item.size < 0
            or name.is_absolute()
            or ".." in name.parts
            or "\\" in item.name
            or any(e["path"] == item.name for e in result["entries"])
        ):
            raise ValueError(
                "unsupported/unsafe/duplicate original TAR member; never skip it"
            )
        result["entries"].append(
            {
                "path": item.name,
                "bytes": item.size,
                "header_offset": offset,
                "data_offset": offset + 512,
                "directory": item.isdir(),
                "part": part["path"],
                "header_sha256": hashlib.sha256(raw).hexdigest(),
                "header_base64": base64.b64encode(raw).decode(),
            }
        )
        result["next_offset"] = offset + 512 + ((item.size + 511) // 512) * 512
    if result["next_offset"] > sum(p["bytes"] for p in parts):
        raise ValueError("captured member exceeds original registered archive")
    return result


def save_owned_checkpoint(directory, index):
    temporary = directory / "checkpoint-next.private.json"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(index, stream)
    os.replace(temporary, directory / "checkpoint-private.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--max-headers", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_headers <= 4096:
        parser.error("explicit header budget must be within1..4096")
    index = empty_index()
    if args.index is not None:
        with args.index.open("rb") as stream:
            raw = stream.read(8 * 1024**2 + 1)
        if len(raw) > 8 * 1024**2:
            parser.error("captured index exceeds byte budget")
        index = json.loads(raw)
    inventory = json.loads(PROFILE.read_text("utf-8"))["media_archive_inventory"]
    selection = prepare(args.data_root)
    if any(
        index.get(key, expected) != expected
        for key, expected in {
            "source_id": "OVOBENCH",
            "dataset_repo": inventory["dataset_repo"],
            "dataset_revision": inventory["dataset_revision"],
        }.items()
    ):
        parser.error("checkpoint publisher revision differs; no cross-source resume")
    report = plan(index, selection, inventory)
    if args.capture:
        if args.output is None or args.output.resolve().is_relative_to(
            PROFILE.parents[2]
        ):
            parser.error("fresh external --output directory required")
        args.output.mkdir(parents=True, exist_ok=False)
        try:
            for step in range(args.max_headers):
                if index["complete"]:
                    break
                index = advance(index, inventory)
                if (step + 1) % 16 == 0 or index["complete"]:
                    save_owned_checkpoint(args.output, index)
        finally:
            # Persist last successful step even when the next network request fails.
            save_owned_checkpoint(args.output, index)
        report = plan(index, selection, inventory)
    print(
        json.dumps(
            {
                "source_id": "OVOBENCH",
                "status": "CAPTURED_HEADER_PROGRESS_NOT_MEDIA_OR_NATIVE_MODEL"
                if args.capture
                else "PLAN_ONLY_NO_NETWORK",
                "captured_header_entries": len(index["entries"]),
                "captured_archive_index_complete": index["complete"],
                "located_original_clips": report["located_original_clips"],
                "required_original_clips": report["required_original_clips"],
                "located_original_bytes": report["located_original_bytes"],
                "model_calls": 0,
                "native_tasks_run": 0,
                "source_checkbox_eligible": False,
            }
        )
    )
