"""Safely gate the unchanged author's full Video-MME archive extraction.

Offline plan by default. Never infer completion from a pre-existing partial
video/subtitle directory; never overwrite it or start a model.
"""

import argparse
import ast
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import zipfile
import uuid

from prepare_videomme_original_assets import MANIFEST, verify_files
from prepare_videomme_original_view import REVISION, ROOT, SOURCE_SHA256, prepare_view

MAX_UNPACKED_BYTES = 250 * 1024**3


def original_unpacker(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unverified original VideoMME source")
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VideoMME"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_dataset"
    )
    functions = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.FunctionDef) and node.name == "unzip_hf_zip"
    ]
    if len(functions) != 1:
        raise ValueError("Original extraction function is ambiguous")
    return compile(
        ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
        "verified-original-videomme-unzip",
        "exec",
    )


def audit_archives(snapshot, descriptors, *, max_unpacked_bytes=MAX_UNPACKED_BYTES):
    """Check original flat-name extraction before any destination is created."""
    result = {"video": {}, "subtitle": {}}
    total = 0
    for item in descriptors:
        if item["role"] != "source_archive":
            continue
        kind = "subtitle" if item["archive_kind"] == "subtitles" else "video"
        with zipfile.ZipFile(Path(snapshot) / item["path"]) as packed:
            entries = packed.infolist()
            if len({entry.filename for entry in entries}) != len(entries):
                raise ValueError("Duplicate original archive member")
            for entry in entries:
                original_name = entry.orig_filename
                path = PurePosixPath(original_name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in original_name
                    or ":" in original_name
                    or "\x00" in original_name
                ):
                    raise ValueError(
                        "Unsafe original archive member retained, not extracted"
                    )
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Original archive symlinks are not supported")
                if entry.is_dir():
                    continue
                if not path.name or path.name in result[kind]:
                    raise ValueError(
                        "Ambiguous native basename overwrite; preserve archives for triage"
                    )
                total += entry.file_size
                if total > max_unpacked_bytes:
                    raise ValueError("Original unpacked byte budget exceeded")
                result[kind][path.name] = {
                    "archive": item["path"],
                    "member": entry.filename,
                    "bytes": entry.file_size,
                    "crc32": entry.CRC,
                }
    return result, total


def verify_extracted(snapshot, layout):
    """Compare every extracted byte with its original archive stream/CRC."""
    identities = []
    groups = {}
    for kind, members in layout.items():
        for basename, item in members.items():
            groups.setdefault(item["archive"], []).append((kind, basename, item))
    for archive, members in groups.items():
        with zipfile.ZipFile(Path(snapshot) / archive) as packed:
            for kind, basename, item in members:
                target = Path(snapshot) / kind / basename
                if not target.is_file() or target.stat().st_size != item["bytes"]:
                    raise ValueError("Incomplete extracted native asset")
                digest = hashlib.sha256()
                with (
                    packed.open(item["member"]) as original,
                    target.open("rb") as actual,
                ):
                    while True:
                        first = original.read(1024 * 1024)
                        second = actual.read(1024 * 1024)
                        if first != second:
                            raise ValueError(
                                "Extracted bytes differ from original archive"
                            )
                        if not first:
                            break
                        digest.update(first)
                identities.append(
                    {"path": f"{kind}/{basename}", **item, "sha256": digest.hexdigest()}
                )
    return identities


def available_memory_bytes():
    """Capacity prerequisite only; no inference/resource attribution."""
    try:
        info = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        available = int(info["MemAvailable"].split()[0]) * 1024
        limit_path = Path("/sys/fs/cgroup/memory.max")
        current_path = Path("/sys/fs/cgroup/memory.current")
        if limit_path.is_file() and current_path.is_file():
            limit = limit_path.read_text().strip()
            if limit != "max":
                current_before = int(current_path.read_text())
                reclaimable = 0
                stats_path = Path("/sys/fs/cgroup/memory.stat")
                if stats_path.is_file():
                    # Count only a conservative lower bound for clean inactive
                    # file cache. Never count active pages, shmem, dirty/writeback
                    # pages, or anonymous model/runtime allocations as free.
                    try:
                        stats = dict(
                            line.split() for line in stats_path.read_text().splitlines()
                        )
                        values = {
                            key: int(stats[key])
                            for key in (
                                "file",
                                "shmem",
                                "inactive_file",
                                "file_dirty",
                                "file_writeback",
                            )
                        }
                        if all(value >= 0 for value in values.values()):
                            reclaimable = max(
                                0,
                                min(
                                    values["inactive_file"],
                                    max(0, values["file"] - values["shmem"]),
                                )
                                - values["file_dirty"]
                                - values["file_writeback"],
                            )
                    except (OSError, KeyError, ValueError):
                        reclaimable = 0
                current_used = max(current_before, int(current_path.read_text()))
                reclaimable = min(reclaimable, max(0, current_used))
                bounded = min(
                    int(limit),
                    max(0, int(limit) - current_used + reclaimable),
                )
                available = min(available, bounded)
        return available
    except (OSError, KeyError, ValueError):
        raise ValueError(
            "Linux memory headroom cannot be verified; no extraction"
        ) from None


@contextmanager
def extraction_lease(snapshot):
    lock = Path(snapshot) / ".videomme-native-extraction.lock"
    nonce = uuid.uuid4().hex
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "owner_pid": os.getpid(),
                "nonce": nonce,
                "purpose": "original-media-extraction-not-model-run",
            },
            stream,
        )
    try:
        yield
    finally:
        # Do not clear an unknown/stale job's lease or any data directory.
        if lock.is_file() and json.loads(lock.read_text())["nonce"] == nonce:
            lock.unlink()


def _prepare_under_lease(checkout, snapshot, output, manifest):
    snapshot = Path(snapshot).resolve(strict=True)
    output = Path(output).resolve()
    if (
        snapshot.name != REVISION
        or snapshot.is_relative_to(ROOT)
        or ROOT.is_relative_to(snapshot)
    ):
        raise ValueError("Pinned external original snapshot required")
    if output.exists() or any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (ROOT, snapshot, Path(checkout).resolve())
    ):
        raise ValueError("Fresh external receipt path separate from sources required")
    if any((snapshot / name).exists() for name in ("video", "subtitle")):
        raise ValueError(
            "Existing native directories retained; never skip or overwrite partial extraction"
        )
    function = original_unpacker(
        (Path(checkout) / "vlmeval/dataset/videomme.py").read_bytes()
    )
    verify_files(snapshot, manifest["files"])
    table = prepare_view(Path(checkout), snapshot)
    with (snapshot / "Video-MME.tsv").open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream, delimiter="\t"))
    required = {row["video"] + ".mp4" for row in records}
    layout, total = audit_archives(snapshot, manifest["files"])
    if (
        len(records) != 2700
        or len(required) != 900
        or not required <= layout["video"].keys()
    ):
        raise ValueError(
            "Required original900 videos missing; no fabricated replacements"
        )
    if sum(name.endswith(".srt") for name in layout["subtitle"]) != 744:
        raise ValueError("Original744 subtitle member set differs")
    if shutil.disk_usage(snapshot).free < total + 100 * 1024**3:
        raise ValueError("Insufficient extraction and service headroom")
    # The unchanged original function reads a complete ZIP member into RAM.
    largest = max(
        item["bytes"] for members in layout.values() for item in members.values()
    )
    if available_memory_bytes() < 3 * largest + 2 * 1024**3:
        raise ValueError(
            "Insufficient memory headroom for unchanged author extraction; do not endanger the model service"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    namespace = {"os": os}
    exec(function, namespace)
    namespace["unzip_hf_zip"](str(snapshot))
    identities = verify_extracted(snapshot, layout)
    result = {
        "source_id": "VIDEOMME",
        "status": "ORIGINAL_FULL_MEDIA_LAYOUT_AND_BYTES_VERIFIED_NOT_NATIVE_INFERENCE",
        "original_tasks": 2700,
        "required_original_videos": 900,
        "original_subtitles": 744,
        "unpacked_bytes": total,
        "model_calls": 0,
        "source_checkbox": False,
        "table": table,
        "extracted_files": identities,
    }
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return {key: value for key, value in result.items() if key != "extracted_files"}


def prepare(checkout, snapshot, output, manifest):
    snapshot = Path(snapshot).resolve(strict=True)
    output = Path(output).resolve()
    if (
        snapshot.name != REVISION
        or snapshot.is_relative_to(ROOT)
        or ROOT.is_relative_to(snapshot)
    ):
        raise ValueError("Pinned external original snapshot required")
    if output.exists() or any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (ROOT, snapshot, Path(checkout).resolve())
    ):
        raise ValueError("Fresh external receipt path separate from sources required")
    with extraction_lease(snapshot):
        return _prepare_under_lease(checkout, snapshot, output, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--accept-academic-only", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "original_tasks": 2700,
                    "model_calls": 0,
                    "downloads": False,
                    "source_checkbox": False,
                }
            )
        )
        return
    if not args.accept_academic_only:
        raise ValueError("Explicit academic-only terms acknowledgement required")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(
        json.dumps(
            prepare(args.official_checkout, args.snapshot, args.output, manifest),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
