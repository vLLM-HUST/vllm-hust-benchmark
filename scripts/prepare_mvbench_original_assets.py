"""Acquire/verify the complete pinned publisher video snapshot, never model tasks.

Offline plan by default. Download requires an explicit byte budget and external
cache. Existing valid assets are reused; raw data never enter this repository.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "workload-preparation/data/catalogs/mvbench_original_video_files_v1.json"
)


def verify_snapshot(snapshot, manifest):
    snapshot = Path(snapshot).resolve(strict=True)
    checked = 0
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in item["path"]:
            raise ValueError("Invalid publisher manifest path")
        path = snapshot / relative
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing or changed original file: {item['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Original file digest differs: {item['path']}")
        checked += 1
    return checked


def ensure_native_ref(cache, manifest):
    """Pin only local Hub branch metadata after original whole-file verification."""
    ref = Path(cache) / "datasets--OpenGVLab--MVBench/refs/video"
    if ref.exists():
        if ref.read_text().strip() != manifest["revision"]:
            raise ValueError("Existing video cache ref differs; no overwrite")
        return
    ref.parent.mkdir(parents=True, exist_ok=True)
    with ref.open("x", encoding="utf-8") as stream:
        stream.write(manifest["revision"])


def materialize(snapshot, data_root, manifest):
    """Make a regular-file view, without copying media or weakening containment."""
    snapshot = Path(snapshot).resolve(strict=True)
    data_root = Path(data_root).resolve()
    if data_root.is_relative_to(ROOT) or ROOT.is_relative_to(data_root):
        raise ValueError("External data view required")
    if data_root.is_relative_to(snapshot) or snapshot.is_relative_to(data_root):
        raise ValueError("Separate fresh data view required")
    data_root.mkdir(parents=True, exist_ok=False)
    for item in manifest["files"]:
        target = data_root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        # Hub snapshots use symlinks to blobs outside the snapshot. The common
        # provider deliberately rejects such escaping links. Hardlink a regular
        # view on the same filesystem; never bypass that boundary or copy15GB.
        os.link((snapshot / item["path"]).resolve(strict=True), target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "verify", "download", "prepare"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--hf-endpoint",
        choices=["https://huggingface.co", "https://hf-mirror.com"],
        default="https://huggingface.co",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    total = sum(item["bytes"] for item in manifest["files"])
    cache = args.cache.resolve()
    snapshot = cache / "datasets--OpenGVLab--MVBench/snapshots" / manifest["revision"]
    if cache.is_relative_to(ROOT) or ROOT.is_relative_to(cache):
        raise ValueError("External cache separate from the repository required")
    result = {
        "source_id": "MVBENCH",
        "status": "PLAN_ONLY",
        "original_files": len(manifest["files"]),
        "original_bytes": total,
        "snapshot": str(snapshot),
        "model_calls": 0,
        "source_checkbox": False,
    }
    if args.action == "plan":
        print(json.dumps(result, indent=2))
        return
    if args.action == "download":
        if args.max_bytes < total:
            raise ValueError("Explicit complete-snapshot byte budget required")
        ancestor = cache
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < 100 * 1024**3:
            raise ValueError("Require at least100GiB free before acquisition")
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=manifest["repo_id"],
                repo_type="dataset",
                revision=manifest["revision"],
                cache_dir=str(cache),
                endpoint=args.hf_endpoint,
                token=False,
                max_workers=3,
            )
        )
    result["verified_files"] = verify_snapshot(snapshot, manifest)
    ensure_native_ref(cache, manifest)
    if args.action == "prepare":
        if args.data_root is None:
            raise ValueError("Explicit external regular-file data root required")
        materialize(snapshot, args.data_root, manifest)
        result["regular_file_view"] = str(args.data_root.resolve())
    result["status"] = "ORIGINAL_FULL_SNAPSHOT_VERIFIED_NOT_NATIVE_EXECUTION"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
