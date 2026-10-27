"""Prepare pinned original RepoBench v1.1 code, not data/models or scoring.

Default is a no-write plan. Raw source stays external. Reuse an existing archive
to avoid downloading again; original source is never patched for NPU execution.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from prepare_labbench_original_assets import extract_original
from run_labbench_original import verify_source

REVISION = "e0cfd34c9e7cd8f057efd32b9280b9719ae12a91"  # pragma: allowlist secret - public author revision
ARCHIVE_SHA256 = "a28a7eede248eee283eda513258dea417801642fc54497df11c99bc4884bd7a5"  # pragma: allowlist secret - public original archive identity
ARCHIVE_BYTES = 1852716
SOURCE_BYTES = 1885511
SOURCE_FILES = 12
URL = f"https://api.github.com/repos/Leolty/repobench/zipball/{REVISION}"


def verify_archive(path):
    if path.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("fixed RepoBench archive size mismatch")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("fixed RepoBench archive checksum mismatch")


def verify_original(source, archive):
    verify_archive(archive)
    return verify_source(
        source, archive, expected_sha=ARCHIVE_SHA256, expected_files=SOURCE_FILES
    )


def prepare(output, *, execute=False, verify_only=False, reuse_archive=None):
    result = {
        "source_id": "REPOBENCH",
        "code_revision": REVISION,
        "url": URL,
        "archive_bytes": ARCHIVE_BYTES,
        "archive_sha256": ARCHIVE_SHA256,
        "source_bytes": SOURCE_BYTES,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if execute and verify_only:
        raise ValueError("select acquisition or read-only verification")
    if not execute and not verify_only:
        return {**result, "status": "PLAN_ONLY"}
    output = Path(output).resolve()
    repo = Path(__file__).resolve().parents[1]
    if (repo / ".git").exists() and output.is_relative_to(repo):
        raise ValueError("publisher assets must remain outside repository")
    archive = output / "original-publisher.zip"
    source = output / "source"
    if verify_only:
        return {
            **result,
            "status": "READ_ONLY_ORIGINAL_SOURCE_VERIFIED",
            "verified_files": verify_original(source, archive),
        }
    if output.exists():
        raise ValueError("fresh output required; inspect and preserve existing files")
    if reuse_archive:
        verify_archive(Path(reuse_archive))
    output.mkdir(parents=True, mode=0o700)
    temporary = output / "original-publisher.zip.partial"
    if reuse_archive:
        try:
            os.link(reuse_archive, temporary)
        except OSError:
            shutil.copyfile(reuse_archive, temporary)
    else:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--max-time",
                "120",
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
    extract_original(
        archive, source, expected_files=SOURCE_FILES, expected_bytes=SOURCE_BYTES
    )
    result.update(
        status="ORIGINAL_CODE_PREPARED_NOT_NATIVE_DELIVERY",
        verified_files=verify_original(source, archive),
    )
    (output / "receipt-public.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-archive", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.output,
                execute=args.execute,
                verify_only=args.verify_only,
                reuse_archive=args.reuse_archive,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
