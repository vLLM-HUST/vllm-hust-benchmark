"""Pinned Video-MME acquisition and annotation-only regular-file preparation.

Offline plan by default. No model execution, frame extraction, task rewriting or
commercial-use permission. Original payloads must remain outside the repository.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "workload-preparation/data/catalogs/videomme_original_release_files_v1.json"
)


def snapshot_path(cache, manifest):
    return (
        Path(cache)
        / ("datasets--" + manifest["repo_id"].replace("/", "--"))
        / "snapshots"
        / manifest["revision"]
    )


def verify_files(snapshot, descriptors):
    count = 0
    for item in descriptors:
        relative = Path(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in item["path"]
            or ":" in item["path"]
        ):
            raise ValueError("Contained publisher file path required")
        path = Path(snapshot) / relative
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Original file missing or size differs: {item['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Original digest differs: {item['path']}")
        count += 1
    return count


def prepare_data_view(snapshot, data_root, manifest):
    """Data only: never pretend videos or full snapshot have been prepared."""
    snapshot = Path(snapshot).resolve(strict=True)
    data_root = Path(data_root).resolve()
    if any(
        data_root.is_relative_to(path) or path.is_relative_to(data_root)
        for path in (ROOT, snapshot)
    ):
        raise ValueError(
            "Fresh external data root separate from repository/snapshot required"
        )
    files = [item for item in manifest["files"] if item["role"] == "data"]
    if len(files) != 1:
        raise ValueError("Exactly one original annotation table required")
    verify_files(snapshot, files)
    data_root.mkdir(parents=True, exist_ok=False)
    for item in files:
        target = data_root / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link((snapshot / item["path"]).resolve(strict=True), target)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--action",
        choices=["plan", "download", "verify", "prepare-data"],
        default="plan",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--accept-academic-only", action="store_true")
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument(
        "--hf-endpoint",
        choices=["https://huggingface.co", "https://hf-mirror.com"],
        default="https://huggingface.co",
    )
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    total = sum(item["bytes"] for item in manifest["files"])
    cache = args.cache.resolve()
    if cache.is_relative_to(ROOT) or ROOT.is_relative_to(cache):
        raise ValueError("External cache separate from repository required")
    snapshot = snapshot_path(cache, manifest)
    result = {
        "source_id": "VIDEOMME",
        "status": "PLAN_ONLY",
        "original_files": len(manifest["files"]),
        "original_release_bytes": total,
        "snapshot": str(snapshot),
        "usage_boundary": manifest["usage_boundary"],
        "model_calls": 0,
        "native_media_ready": False,
        "source_checkbox": False,
    }
    if args.action == "plan":
        print(json.dumps(result, indent=2))
        return
    if not args.accept_academic_only:
        raise ValueError("Explicit academic-only terms acknowledgement required")
    if args.action == "download":
        if args.max_bytes < total:
            raise ValueError("Explicit complete original release byte budget required")
        ancestor = cache
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < 500 * 1024**3:
            raise ValueError(
                "Reserve at least500GiB for original archives, extraction and service headroom"
            )
        os.environ.update(
            HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
            HF_HUB_DISABLE_XET="1",
            HF_HUB_DOWNLOAD_TIMEOUT="60",
        )
        from huggingface_hub import snapshot_download

        snapshot = Path(
            snapshot_download(
                repo_id=manifest["repo_id"],
                repo_type="dataset",
                revision=manifest["revision"],
                cache_dir=str(cache),
                endpoint=args.hf_endpoint,
                token=False,
                max_workers=2,
            )
        )
        if snapshot.name != manifest["revision"]:
            raise ValueError("Original pinned snapshot revision differs")
    if args.action == "prepare-data":
        if args.data_root is None:
            raise ValueError("Explicit external annotation data root required")
        result["verified_annotation_files"] = prepare_data_view(
            snapshot, args.data_root, manifest
        )
        result["status"] = "ANNOTATION_ONLY_REGULAR_VIEW_NOT_MEDIA_OR_NATIVE_EXECUTION"
    else:
        result["verified_files"] = verify_files(snapshot, manifest["files"])
        result["status"] = (
            "COMPLETE_ORIGINAL_ARCHIVES_VERIFIED_NOT_EXTRACTED_OR_EXECUTED"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
