"""Prepare the entire fixed official probe archive; no models or task rewriting."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT
    / "workload-preparation/data/catalogs/temporalwiki_original_probe_files_v1.json"
)
ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")


def verify_bytes(raw, spec):
    if len(raw) != spec["bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("original publisher asset identity mismatch")


def verify_archive(path, profile):
    """Check every original member and CSV row; never extract or execute code."""
    if Path(path).stat().st_size != profile["archive"]["bytes"]:
        raise ValueError("original archive size differs")
    verify_bytes(Path(path).read_bytes(), profile["archive"])
    expected = {item["path"]: item for item in profile["files"]}
    directories = {str(PurePosixPath(name).parent) + "/" for name in expected}
    if len(expected) != len(profile["files"]):
        raise ValueError("distinct registered archive members required")
    with zipfile.ZipFile(path) as packed:
        entries = packed.infolist()
        if len({item.filename for item in entries}) != len(entries):
            raise ValueError("duplicate original archive member")
        if {item.filename for item in entries} != set(expected) | directories:
            raise ValueError("original archive member set differs")
        if sum(item.file_size for item in entries) > 64 * 1024 * 1024:
            raise ValueError("probe archive exceeds fixed unpacked safety budget")
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in entry.filename:
                raise ValueError("contained original archive members required")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("archive symlinks are not supported")
            if entry.is_dir():
                continue
            spec = expected[entry.filename]
            raw = packed.read(entry)
            verify_bytes(raw, spec)
            if spec["role"] == "data":
                reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
                if reader.fieldnames != spec["columns"]:
                    raise ValueError("original probe columns differ")
                count = 0
                for row in reader:
                    if None in row or None in row.values():
                        raise ValueError("original probe CSV width differs")
                    count += 1
                if count != spec["rows"]:
                    raise ValueError("original probe row count differs")
    return len(expected)


def verify_data(root, profile):
    root = Path(root).resolve(strict=True)
    expected = {item["path"] for item in profile["files"]}
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise ValueError("extracted original file set differs")
    for item in profile["files"]:
        path = (root / item["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("original file escapes explicit data root")
        verify_bytes(path.read_bytes(), item)


def extract_archive(archive, root, profile):
    verify_archive(archive, profile)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    with zipfile.ZipFile(archive) as packed:
        for item in profile["files"]:
            path = (root / item["path"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("extraction escapes explicit data root")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(packed.read(item["path"]))
    verify_data(root, profile)


def download_archive(path, profile, endpoint, max_bytes):
    if endpoint not in ENDPOINTS or max_bytes < profile["archive"]["bytes"]:
        raise ValueError(
            "explicit supported endpoint and complete byte budget required"
        )
    path = Path(path)
    if path.exists():
        verify_archive(path, profile)
        return
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ValueError("inspect existing partial; never overwrite it")
    url = f"{endpoint}/datasets/{profile['dataset_repo']}/resolve/{profile['dataset_revision']}/{profile['archive']['path']}?download=true"
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(
        Request(
            url,
            headers={
                "User-Agent": "huggingface_hub/0.36.0",
                "Accept-Encoding": "identity",
            },
        ),
        timeout=60,
    ) as response:
        with partial.open("xb") as stream:
            raw = response.read(profile["archive"]["bytes"] + 1)
            stream.write(raw)
    verify_archive(partial, profile)
    os.link(partial, path)
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--action", choices=("plan", "download", "extract", "verify"), default="plan"
    )
    parser.add_argument("--hf-endpoint", choices=ENDPOINTS, default=ENDPOINTS[0])
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text("utf-8"))
    for path in (args.archive, args.data_root):
        if path is not None and path.resolve().is_relative_to(ROOT):
            parser.error("original raw assets must stay outside the repository")
    if args.action == "download":
        if args.max_bytes is None:
            parser.error("explicit complete download byte budget required")
        download_archive(args.archive, profile, args.hf_endpoint, args.max_bytes)
    elif args.action == "extract":
        if args.data_root is None:
            parser.error("fresh external --data-root required")
        extract_archive(args.archive, args.data_root, profile)
    elif args.action == "verify":
        verify_archive(args.archive, profile)
        if args.data_root is not None:
            verify_data(args.data_root, profile)
    print(
        json.dumps(
            {
                "source_id": "TEMPORALWIKI",
                "action": args.action,
                "status": "PLAN_ONLY_NO_DOWNLOAD"
                if args.action == "plan"
                else "ORIGINAL_PROBE_ASSETS_VERIFIED_NOT_NATIVE_EXECUTION",
                "files": len(profile["files"]),
                "original_probe_rows": sum(
                    item.get("rows", 0) for item in profile["files"]
                ),
                "model_calls": 0,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
