"""Prepare a registered fixed original clip, never a whole archive or model run.

Default is a plan. Explicit download requires a captured original header index,
sealed original annotations, registered complete-member identity and byte budget.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import shutil
from urllib.request import urlopen

from plan_ovobench_original_media_ranges import PROFILE, plan
from prepare_ovobench_native_selection import prepare
from prepare_wavcaps_original_media import fetch
from vllm_hust_workload_data.assets import verify_file


def transfer(member, location, inventory, etags, sink, *, opener=urlopen):
    parts = {item["path"]: item for item in inventory["parts"]}
    if (
        location["path"] != member["path"]
        or location["bytes"] != member["bytes"]
        or location["archive_member"] != member["archive_member"]
        or sum(item["end"] - item["start"] + 1 for item in location["ranges"])
        != member["bytes"]
    ):
        raise ValueError("complete registered original member span required")
    digest, count = hashlib.sha256(), 0
    for span in location["ranges"]:
        part = parts[span["part"]]
        expected_etag = etags.get(span["part"])
        if not expected_etag or not 0 <= span["start"] <= span["end"] < part["bytes"]:
            raise ValueError("original part bounds and captured ETag required")
        start = span["start"]
        while start <= span["end"]:
            end = min(start + 8 * 1024**2 - 1, span["end"])
            item = {
                "url": f"https://huggingface.co/datasets/{inventory['dataset_repo']}/resolve/{inventory['dataset_revision']}/{part['path']}?member_range={start}",
                "start": start,
                "end": end,
                "part_bytes": part["bytes"],
            }

            def checked_open(request, *, timeout):
                response = opener(request, timeout=timeout)
                if response.headers.get("ETag") != expected_etag:
                    response.close()
                    raise ValueError(
                        "publisher part differs from captured original ETag"
                    )
                return response

            raw = fetch(item, opener=checked_open)
            if sink.write(raw) != len(raw):
                raise OSError("incomplete original media write")
            digest.update(raw)
            count += len(raw)
            start = end + 1
    if count != member["bytes"] or digest.hexdigest() != member["sha256"]:
        raise ValueError(
            "complete original member identity mismatch; do not use partial media"
        )
    return {"path": member["path"], "bytes": count, "sha256": digest.hexdigest()}


def download(member, location, inventory, etags, media_root, *, max_bytes):
    name = member["path"]
    if Path(name).name != name or "\\" in name or PureWindowsPath(name).drive:
        raise ValueError("registered original clip basename required")
    root = Path(media_root).resolve()
    if root.is_relative_to(PROFILE.parents[2]):
        raise ValueError("raw media must stay outside the repository")
    if max_bytes < member["bytes"]:
        raise ValueError("explicit complete original member byte budget required")
    root.mkdir(parents=True, exist_ok=True)
    target = root / name
    if target.exists():
        return {
            **verify_file(root, member),
            "existing_original_reused": True,
            "network_fetch_executed": False,
        }
    partial = target.with_name(target.name + ".partial")
    if partial.exists():
        raise ValueError("inspect existing partial file; never overwrite it")
    if shutil.disk_usage(root).free < member["bytes"] + 512 * 1024**2:
        raise ValueError("insufficient media space including 512MiB reserve")
    with partial.open("xb") as sink:
        result = transfer(member, location, inventory, etags, sink)
    verify_file(root, {**member, "path": partial.name})
    os.link(partial, target)
    partial.unlink()
    return {**result, "existing_original_reused": False, "network_fetch_executed": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--member", required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--max-bytes", type=int, default=0)
    args = parser.parse_args()
    with args.index.open("rb") as stream:
        raw = stream.read(8 * 1024**2 + 1)
    if len(raw) > 8 * 1024**2:
        parser.error("captured index exceeds byte budget")
    index = json.loads(raw)
    inventory = json.loads(PROFILE.read_text("utf-8"))["media_archive_inventory"]
    members = {item["path"]: item for item in inventory["qualified_original_members"]}
    if args.member not in members:
        parser.error(
            "original member not yet identity-qualified; do not substitute another clip"
        )
    member = members[args.member]
    planned = plan(index, prepare(args.data_root), inventory)
    location = next(
        (item for item in planned["members"] if item["path"] == args.member), None
    )
    if location is None:
        parser.error("fixed original member not located in captured header index")
    result = {
        "member": member["path"],
        "bytes": member["bytes"],
        "ranges": location["ranges"],
    }
    if args.download:
        if args.media_root is None:
            parser.error("explicit external --media-root required")
        result = download(
            member,
            location,
            inventory,
            index["etags"],
            args.media_root,
            max_bytes=args.max_bytes,
        )
    print(
        json.dumps(
            {
                "source_id": "OVOBENCH",
                "status": "REGISTERED_ORIGINAL_MEDIA_PREPARATION_NOT_NATIVE_MODEL"
                if args.download
                else "PLAN_ONLY_NO_NETWORK",
                **result,
                "model_calls": 0,
                "native_tasks_run": 0,
                "source_checkbox_eligible": False,
            }
        )
    )
