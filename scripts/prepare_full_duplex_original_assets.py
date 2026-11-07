"""Prepare unchanged official audio archives externally; never run speech models."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs/workload-data/dataset-validation-status.json"


def load_pins(path=LEDGER):
    source = json.loads(Path(path).read_text("utf-8"))["delivery_inventory"][
        "FULL_DUPLEX_BENCH"
    ]
    acquisition = source["original_archive_acquisition"]
    hashes = acquisition["observed_archive_sha256"]
    rows = source["publisher_dataset_directory_inventory"]["archives"]
    result = []
    for row in rows:
        name = row["name"]
        if (
            row["version"] not in {"v1.0", "v1.5"}
            or re.fullmatch(r"[a-z_]+[.]zip", name) is None
            or re.fullmatch(r"[A-Za-z0-9_-]{20,80}", row["file_id"]) is None
            or type(row["reported_bytes"]) is not int
            or not 0 < row["reported_bytes"] <= 200 * 1024**2
            or re.fullmatch(r"[0-9a-f]{64}", hashes[name]) is None
        ):
            raise ValueError("contained official archive identity required")
        result.append(
            {
                "path": row["version"] + "/" + name,
                "bytes": row["reported_bytes"],
                "sha256": hashes[name],
                "url": "https://drive.usercontent.google.com/download?id="
                + row["file_id"]
                + "&export=download&confirm=t",
            }
        )
    if (
        len(result) != 9
        or len({x["path"] for x in result}) != 9
        or set(hashes) != {x["name"] for x in rows}
        or sum(x["bytes"] for x in result) != 705372348
    ):
        raise ValueError("complete frozen nine-archive selection required")
    return result


def archive_path(root, name):
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or ":" in name
        or relative.as_posix() != name
    ):
        raise ValueError("contained archive path required")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("archive escapes output root")
    for candidate in [path, *path.parents]:
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError("archive symlink not allowed")
    return path


def verify_archive(path, item):
    if path.is_symlink() or path.stat().st_size != item["bytes"]:
        raise ValueError("original archive size or file identity differs")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != item["sha256"]:
        raise ValueError("frozen observed original archive SHA256 differs")
    with zipfile.ZipFile(path) as archive:
        files = archive.infolist()
        if sum(x.file_size for x in files) > 1024**3:
            raise ValueError("original archive uncompressed budget exceeded")
        names = set()
        for entry in files:
            stored_name = entry.orig_filename
            name = PurePosixPath(stored_name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in stored_name
                or ":" in stored_name
                or "\x00" in stored_name
                or stored_name != entry.filename
                or entry.filename in names
                or (entry.external_attr >> 16) & 0o170000 == 0o120000
                or entry.flag_bits & 1
            ):
                raise ValueError("unsafe or ambiguous original ZIP member")
            names.add(entry.filename)
        if archive.testzip() is not None:
            raise ValueError("original ZIP CRC differs")
        json_files = [
            x
            for x in files
            if not x.is_dir()
            and x.filename.endswith(".json")
            and not x.filename.startswith("__MACOSX/")
        ]
        for entry in json_files:
            if entry.file_size > 8 * 1024**2:
                raise ValueError("original JSON verification budget exceeded")
            json.loads(archive.read(entry))
    return {"path": item["path"], "original_json_decoded": len(json_files)}


def prepare(root, items, *, opener=urlopen):
    for item in items:
        path = archive_path(root, item["path"])
        if path.exists():
            verify_archive(path, item)
            continue
        partial = path.with_name(path.name + ".partial")
        if partial.exists():
            raise ValueError("inspect retained partial before retry; no overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        with opener(item["url"], timeout=90) as response, partial.open("xb") as stream:
            remaining = item["bytes"]
            while True:
                block = response.read(min(1024**2, remaining + 1))
                if not block:
                    break
                if len(block) > remaining:
                    raise ValueError("publisher response exceeds pinned byte budget")
                stream.write(block)
                remaining -= len(block)
        verify_archive(partial, item)
        os.link(partial, path)
        partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("original archives must remain outside checkout")
    items = load_pins()
    if args.action == "prepare":
        if args.max_bytes is None or args.max_bytes < sum(x["bytes"] for x in items):
            raise ValueError("explicit budget must cover all original archives")
        prepare(root, items)
    verified = (
        [verify_archive(archive_path(root, x["path"]), x) for x in items]
        if args.action != "plan"
        else []
    )
    print(
        json.dumps(
            {
                "source_id": "FULL_DUPLEX_BENCH",
                "status": "PLAN_ONLY" if not verified else "ORIGINAL_ARCHIVES_VERIFIED",
                "archives": len(items),
                "bytes": sum(x["bytes"] for x in items),
                "verified": verified,
                "hash_scope": "observed official download hashes,not publisher checksums",
                "archive_extraction_performed": False,
                "model_calls": 0,
                "source_checkbox_eligible": False,
            }
        )
    )


if __name__ == "__main__":
    main()
