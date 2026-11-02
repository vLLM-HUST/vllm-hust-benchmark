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
import subprocess
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


def decode_command(ffmpeg, media):
    return [
        str(ffmpeg),
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-xerror",
        "-threads",
        "2",
        "-i",
        str(media),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vsync",
        "0",
        "-progress",
        "pipe:1",
        "-nostats",
        "-f",
        "null",
        "-",
    ]


def decoded_frames(progress):
    values = dict(line.split("=", 1) for line in progress.splitlines() if "=" in line)
    if values.get("progress") != "end" or int(values.get("frame", "0")) <= 0:
        raise ValueError("decoder did not reach a positive-frame terminal end")
    return int(values["frame"])


def verify_media(profile, args):
    root = args.verify_media_root.resolve(strict=True)
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.is_relative_to(root)
        or root.is_relative_to(output)
        or output.is_relative_to(repository)
    ):
        raise ValueError(
            "fresh audit output outside original media and repository required"
        )
    saved = json.loads((root / "media-public.json").read_bytes())
    if (
        saved.get("source_id") != "VIDEOMME_V2"
        or saved.get("archive_sha256") != profile["archive_sha256"]
        or saved.get("selected_video_ids") != profile["selected_video_ids"]
        or [item["video_id"] for item in saved.get("files", [])]
        != profile["selected_video_ids"]
    ):
        raise ValueError(
            "saved original archive/member binding differs from frozen selection"
        )
    for item in saved["files"]:
        path = (root / item["path"]).resolve(strict=True)
        if (
            path.parent != root
            or path.name != item["video_id"] + ".mp4"
            or path.stat().st_size != item["bytes"]
            or digest(path) != item["observed_sha256"]
        ):
            raise ValueError("saved original media identity differs")
    software = {}
    if args.decode:
        ffmpeg = args.ffmpeg.resolve(strict=True)
        if ffmpeg.is_relative_to(root):
            raise ValueError("trusted decoder must not come from dataset directory")
        result = subprocess.run(
            [str(ffmpeg), "-version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
        software = {
            "ffmpeg_sha256": digest(ffmpeg),
            "ffmpeg_version": result.stdout.splitlines()[0],
        }
    output.mkdir(mode=0o700, exist_ok=False)
    report = {
        "source_id": "VIDEOMME_V2",
        "status": "SAVED_ORIGINAL_MEDIA_HASHES_VERIFIED_NOT_DECODED",
        "selected_video_ids": profile["selected_video_ids"],
        "files": [],
        "decoder": software,
        "video_stream_only_audio_retained_not_decoded": True,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    try:
        for item in saved["files"]:
            record = {
                "video_id": item["video_id"],
                "original_media_sha256": item["observed_sha256"],
            }
            if args.decode:
                result = subprocess.run(
                    decode_command(ffmpeg, root / item["path"]),
                    capture_output=True,
                    text=True,
                    timeout=args.decoder_timeout,
                )
                (output / (item["video_id"] + "-decoder-stdout.txt")).write_text(
                    result.stdout, encoding="utf-8"
                )
                (output / (item["video_id"] + "-decoder-stderr.txt")).write_text(
                    result.stderr, encoding="utf-8"
                )
                if result.returncode:
                    raise RuntimeError(
                        "original video decoder failed; retain logs and do not replace media"
                    )
                record["all_video_frames_decoded"] = decoded_frames(result.stdout)
                if digest(root / item["path"]) != item["observed_sha256"]:
                    raise ValueError("original video changed during decode")
            report["files"].append(record)
            (output / "progress-public.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
        if args.decode:
            report["status"] = (
                "ALL_FIXED_ORIGINAL_VIDEO_FRAMES_DECODED_NOT_NATIVE_INFERENCE"
            )
        (output / "qualification-public.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report.update(
            status="MEDIA_AUDIT_FAILED_RETAIN_ORIGINALS_NO_REPLACEMENT",
            error_type=type(error).__name__,
            error=str(error),
        )
        (output / "failure-public.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--verify-media-root", type=Path)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--decoder-timeout", type=float, default=1800)
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
    if (
        (args.archive is None) == (args.verify_media_root is None)
        or args.output is None
        or args.storage_budget_bytes <= 0
        or args.decoder_timeout <= 0
        or (args.decode and (args.verify_media_root is None or args.ffmpeg is None))
    ):
        parser.error(
            "one archive or saved media root, fresh output and explicit decoder for decode mode required"
        )
    os.umask(0o077)
    result = (
        verify_media(profile, args)
        if args.verify_media_root
        else extract(
            profile, args.archive, args.output, budget=args.storage_budget_bytes
        )
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
