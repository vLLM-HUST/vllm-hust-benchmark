"""Build standalone custody from an inventoried server snapshot; never edit sources."""

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TAG = "evidence-frontier-recovered-20261021-v1"
ASSET = "frontier-recovered-receipts-20261021.tar.gz"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe inventory path")
    return path


def build(inventory_path, destination, index_path):
    inventory = json.loads(inventory_path.read_text())
    index = json.loads(index_path.read_text())
    if destination.exists():
        raise ValueError("Refusing to replace a custody snapshot")
    source = Path(inventory["source_root"]).resolve()
    known = {}
    for asset in index["assets"]:
        for member in asset["files"]:
            known.setdefault(
                member["sha256"],
                {
                    "archive_url": asset["url"],
                    "archive_sha256": asset["sha256"],
                    "member": member["member"],
                },
            )
    rows = []
    seen = set()
    members = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for record in inventory["files"]:
            name = str(safe_path(record["original_path"]))
            if name in seen:
                raise ValueError("Duplicate inventory path")
            seen.add(name)
            path = source / name
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise ValueError("Source escaped inventory root")
            raw = path.read_bytes()
            identity = sha(raw)
            if identity != record["sha256"] or len(raw) != record["bytes"]:
                raise ValueError(f"Source changed since inventory: {name}")
            if identity not in known:
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(raw))
                members.append({"member": name, "sha256": identity, "bytes": len(raw)})
                known[identity] = {"asset": ASSET, "member": name}
            row = {
                "original_path": name,
                "sha256": identity,
                "bytes": len(raw),
                "source": known[identity],
            }
            if "run_status" in record:
                row["run_status"] = record["run_status"]
            rows.append(row)
    compressed = gzip.compress(buffer.getvalue(), mtime=0)
    archive_sha = sha(compressed)
    archive_url = (
        "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
        f"{TAG}/{ASSET}"
    )
    for row in rows:
        if "asset" in row["source"]:
            row["source"] = {
                "archive_url": archive_url,
                "archive_sha256": archive_sha,
                "member": row["source"]["member"],
            }
    manifest = {
        "schema": "recovered-receipt-custody/v1",
        "release_tag": TAG,
        "source_root": str(source),
        "inventory_sha256": sha(inventory_path.read_bytes()),
        "reused_index_sha256": sha(index_path.read_bytes()),
        "selection": inventory["selection"],
        "limitations": inventory["limitations"],
        "excluded_directories": inventory["excluded_directories"],
        "included_suffixes": inventory["included_suffixes"],
        "asset": {
            "name": ASSET,
            "url": archive_url,
            "bytes": len(compressed),
            "sha256": archive_sha,
            "members": members,
        },
        "files": rows,
        "admission": "Custody only. No new performance claim, pairing or qualification.",
        "raw_source_unchanged": True,
    }
    destination.mkdir(parents=True, mode=0o700)
    (destination / ASSET).write_bytes(compressed)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "paths": len(rows),
                "new_members": len(members),
                "archive_bytes": len(compressed),
                "invalid_statuses": sum(
                    "parse_error" in row.get("run_status", {}) for row in rows
                ),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "archive/frontier-20260927-v1/manifest.json",
    )
    args = parser.parse_args()
    build(args.inventory, args.destination, args.index)
