"""Verify server-private original selected assets; never download or execute.

Default plan-only. Fixed task selection and identities were acquired from the
publisher's verified archive. A passed check is not native benchmark delivery.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zlib

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/scienceagentbench-original-fixed3-assets.json"
)


def verify_file(root, item):
    root = Path(root).resolve(strict=True)
    relative = Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("contained original asset path required")
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError("asset escapes original root")
    if not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError("original asset missing or size mismatch")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != item["sha256"]:
        raise ValueError("original asset SHA256 mismatch")
    if "original_crc32" in item and zlib.crc32(raw) != item["original_crc32"]:
        raise ValueError("original ZIP member CRC mismatch")


def verify(metadata_root, asset_root):
    profile = json.loads(PROFILE.read_text("utf-8"))
    verify_file(metadata_root, profile["metadata"])
    for item in profile["files"]:
        verify_file(asset_root, item)
    root = Path(asset_root).resolve(strict=True)
    for relative in profile["complete_dataset_roots"]:
        folder = root / relative
        if folder.is_symlink() or not folder.resolve().is_relative_to(root):
            raise ValueError("original dataset directory escapes asset root")
        all_paths = list(folder.rglob("*"))
        if any(p.is_symlink() for p in all_paths):
            raise ValueError("unexpected link inside original dataset directory")
        actual = {p.relative_to(root).as_posix() for p in all_paths if p.is_file()}
        expected = {
            x["path"] for x in profile["files"] if x["path"].startswith(relative + "/")
        }
        if actual != expected:
            raise ValueError("original dataset folder contains missing or extra files")
    return {
        "source_id": "SCIENCEAGENTBENCH",
        "status": "ORIGINAL_FIXED3_ASSETS_VERIFIED_NOT_NATIVE_EXECUTION",
        "original_instance_ids": profile["instance_ids"],
        "verified_metadata_files": 1,
        "verified_asset_files": len(profile["files"]),
        "verified_asset_bytes": sum(x["bytes"] for x in profile["files"]),
        "complete_original_dataset_directories": len(profile["complete_dataset_roots"]),
        "model_calls": 0,
        "generated_or_reference_code_executed": False,
        "source_checkbox_eligible": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--action", choices=["plan", "verify"], default="plan")
    args = parser.parse_args(argv)
    result = (
        verify(args.metadata_root, args.asset_root)
        if args.action == "verify"
        else {
            "source_id": "SCIENCEAGENTBENCH",
            "status": "PLAN_ONLY",
            "model_calls": 0,
            "source_checkbox_eligible": False,
        }
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
