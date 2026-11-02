"""Verify original checkpoint bytes/ZIP without deserialization or model calls."""

import argparse
import json
from pathlib import Path, PurePosixPath
import zipfile

from verify_dailytalk_original_models import verify_file

PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/wavcaps-original-captioning-checkpoint.json"
)


def verify(path, profile):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular original checkpoint file required")
    expected = profile["checkpoint"]
    verify_file(path, expected)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if (
            len(members) != expected["zip_members"]
            or sum(item.file_size for item in members) > 8 * 1024**3
        ):
            raise ValueError("original checkpoint ZIP structure differs")
        for item in members:
            name = PurePosixPath(item.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in item.filename
                or ((item.external_attr >> 16) & 0o170000) == 0o120000
            ):
                raise ValueError("unsafe original checkpoint ZIP member")
        if archive.testzip() is not None:
            raise ValueError("original checkpoint ZIP CRC differs")
    return len(members)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--action", choices=["plan", "verify"], default="plan")
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text("utf-8"))
    report = {
        "source_id": "WAVCAPS",
        "status": "PLAN_ONLY_NO_MODEL_LOAD",
        "checkpoint": profile["checkpoint"],
        "model_files_deserialized": False,
        "model_calls": 0,
        "npu_forward_qualified": False,
        "source_checkbox_eligible": False,
    }
    if args.action == "verify":
        if args.checkpoint is None:
            parser.error("--checkpoint is required for verification")
        report["verified_zip_members"] = verify(args.checkpoint, profile)
        report["status"] = "ORIGINAL_CHECKPOINT_VERIFIED_NOT_NATIVE_EXECUTION"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
