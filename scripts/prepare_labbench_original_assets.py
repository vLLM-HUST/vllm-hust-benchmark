"""Plan/acquire/verify the unchanged fixed LAB-Bench publisher repository.

Default is plan-only. Original tasks and images stay in an explicit external
directory; this command never installs packages or invokes a model.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
from zipfile import ZipFile

from run_labbench_original import ARCHIVE_SHA256, REVISION, verify_source

ARCHIVE_BYTES = 245275217
SOURCE_BYTES = 283413040
URL = f"https://api.github.com/repos/Future-House/LAB-Bench/zipball/{REVISION}"


def extract_original(
    archive, source, *, expected_files=553, expected_bytes=SOURCE_BYTES
):
    source = Path(source).resolve()
    if source.exists():
        raise ValueError("fresh source output required")
    with ZipFile(archive) as zipped:
        entries = [item for item in zipped.infolist() if not item.is_dir()]
        if (
            len(entries) != expected_files
            or sum(item.file_size for item in entries) != expected_bytes
        ):
            raise ValueError("original source inventory mismatch")
        prefix = entries[0].filename.split("/")[0] + "/"
        names = set()
        for item in entries:
            name = item.filename.removeprefix(prefix)
            relative = PurePosixPath(name)
            if (
                not item.filename.startswith(prefix)
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in item.orig_filename
                or name in names
                or not (source / name).resolve().is_relative_to(source)
                or (item.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError("unsafe or duplicate archive member")
            names.add(name)
        source.mkdir(mode=0o700, parents=True)
        for item in entries:
            target = source / item.filename[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(zipped.read(item))
    return len(entries)


def verify_archive(path):
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("fixed publisher archive size mismatch")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("fixed publisher archive checksum mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-archive", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = {
        "source_id": "LABBENCH",
        "revision": REVISION,
        "url": URL,
        "archive_bytes": ARCHIVE_BYTES,
        "source_bytes": SOURCE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if not args.execute and not args.verify_only:
        print(json.dumps({**result, "status": "PLAN_ONLY"}))
        return
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if (repository / ".git").exists() and output.is_relative_to(repository):
        raise ValueError("raw assets must stay outside this repository")
    archive = output / "publisher-original.zip"
    source = output / "source"
    if args.verify_only:
        verify_archive(archive)
        count = verify_source(source, archive)
        print(
            json.dumps(
                {
                    **result,
                    "status": "READ_ONLY_ORIGINAL_ASSETS_VERIFIED",
                    "verified_files": count,
                }
            )
        )
        return
    if output.exists():
        raise ValueError(
            "fresh external output required; inspect partial results first"
        )
    if args.reuse_archive:
        verify_archive(args.reuse_archive)
    output.mkdir(mode=0o700, parents=True)
    temporary = output / "publisher-original.zip.partial"
    if args.reuse_archive:
        try:
            os.link(args.reuse_archive, temporary)
        except OSError:
            shutil.copyfile(args.reuse_archive, temporary)
    else:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--max-time",
                "600",
                "--max-filesize",
                str(ARCHIVE_BYTES),
                "--output",
                str(temporary),
                URL,
            ],
            check=True,
        )
    verify_archive(temporary)
    temporary.rename(archive)
    extract_original(archive, source)
    count = verify_source(source, archive)
    result.update(
        status="ORIGINAL_ASSETS_PREPARED_NOT_NATIVE_DELIVERY", verified_files=count
    )
    (output / "receipt-public.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
