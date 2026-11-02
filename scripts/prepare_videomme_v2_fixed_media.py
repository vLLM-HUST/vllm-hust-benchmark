"""Verify an original Video-MME-v2 archive and extract frozen complete-group media.

Default prints the original download plan. --execute reads an existing archive;
it never downloads, transcodes, removes audio, runs models or changes questions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/videomme-v2-original-fixed-media.json"
)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def selected_members(archive, identifiers):
    if len(set(identifiers)) != len(identifiers) or not identifiers:
        raise ValueError("distinct frozen video IDs required")
    result = []
    for identity in identifiers:
        if len(identity) != 3 or not identity.isascii() or not identity.isdigit():
            raise ValueError("original three-digit video ID required")
        matches = [
            item
            for item in archive.infolist()
            if not item.is_dir()
            and PurePosixPath(item.filename).name == identity + ".mp4"
        ]
        if len(matches) != 1:
            raise ValueError(
                "original selected video missing or ambiguous; do not replace it"
            )
        item = matches[0]
        name = PurePosixPath(item.orig_filename)
        kind = stat.S_IFMT(item.external_attr >> 16)
        if (
            name.is_absolute()
            or ".." in name.parts
            or "\\" in item.orig_filename
            or kind not in (0, stat.S_IFREG)
            or item.flag_bits & 1
        ):
            raise ValueError("unsafe or encrypted original ZIP member")
        result.append(item)
    return result


def extract(profile, archive_path, output, *, budget):
    archive_path = archive_path.resolve(strict=True)
    output = output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.is_relative_to(repository)
        or archive_path.is_relative_to(output)
    ):
        raise ValueError(
            "fresh output outside repository and existing archive required"
        )
    if (
        archive_path.stat().st_size != profile["archive_bytes"]
        or digest(archive_path) != profile["archive_sha256"]
    ):
        raise ValueError("original whole-archive size/SHA256 differs")
    with zipfile.ZipFile(archive_path) as archive:
        members = selected_members(archive, profile["selected_video_ids"])
        required = sum(item.file_size for item in members)
        if (
            required > budget
            or shutil.disk_usage(output.parent).free < required + 256 * 1024**2
        ):
            raise ValueError("insufficient explicit media budget or disk space")
        output.mkdir(mode=0o700, exist_ok=False)
        records = []
        for item in members:
            target = output / PurePosixPath(item.filename).name
            # Reading to EOF verifies the publisher archive's member CRC.
            with archive.open(item) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
            if target.stat().st_size != item.file_size:
                raise ValueError("original media extraction size mismatch")
            records.append(
                {
                    "original_zip_member": item.filename,
                    "video_id": target.stem,
                    "path": target.name,
                    "bytes": item.file_size,
                    "original_crc32": item.CRC,
                    "observed_sha256": digest(target),
                }
            )
    report = {
        "source_id": "VIDEOMME_V2",
        "status": "FIXED_ORIGINAL_MEDIA_EXTRACTED_NOT_DECODED_OR_EXECUTED",
        "archive_sha256": profile["archive_sha256"],
        "selected_video_ids": profile["selected_video_ids"],
        "files": records,
        "all_selected_member_crc_verified": True,
        "transcoded_or_audio_removed": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    (output / "media-public.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--storage-budget-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    profile = json.loads(PROFILE.read_text("utf-8"))
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "original_url": profile["url"],
                    "archive_bytes": profile["archive_bytes"],
                    "archive_sha256": profile["archive_sha256"],
                    "selected_video_ids": profile["selected_video_ids"],
                    "model_calls": 0,
                }
            )
        )
        return
    if args.archive is None or args.output is None or args.storage_budget_bytes <= 0:
        parser.error(
            "original archive, fresh output and positive storage budget required"
        )
    os.umask(0o077)
    print(
        json.dumps(
            extract(
                profile, args.archive, args.output, budget=args.storage_budget_bytes
            )
        )
    )


if __name__ == "__main__":
    main()
