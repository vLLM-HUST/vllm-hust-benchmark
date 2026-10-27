"""Prepare unchanged Copilot 2026 release metadata outside the repository.

Default is a no-write plan. Seven official archives are pinned independently
of the later publisher documentation revision. No prompts, replay, analysis,
model calls or synthetic sessions are produced.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from vllm_hust_workload_data.copilot import (
    extract_verified_archive,
    verify_original_directory,
)

RELEASE_TAG = "ghcp-coding-agent-2026"
RELEASE_ID = 400522856
ARCHIVES = [
    # Public release asset IDs and checksums, never credentials.
    (
        "2026-06-01",
        602046479,
        594479134,
        "cacad69f23d3c87ac54d788fec3f8e9c84df82c0cc9638d693340423c14e68e9",  # pragma: allowlist secret - public release SHA256
    ),  # pragma: allowlist secret
    (
        "2026-06-02",
        602046803,
        621345734,
        "95b0450041095dbe88181b88841c64b1e42f55a0127aa7abe91c1c9d3b424240",  # pragma: allowlist secret - public release SHA256
    ),  # pragma: allowlist secret
    (
        "2026-06-03",
        602047067,
        618403924,
        "622a5b0742eeb196d69d9a5ee5f6457cf82d432ddc435b9eb76eb5bb4363f1d0",  # pragma: allowlist secret - public release SHA256
    ),  # pragma: allowlist secret
    (
        "2026-06-04",
        602047355,
        572771629,
        "fd81d45e0149cae10aa193f2c2536ae709a5eba61b5aa886e534587e80473028",  # pragma: allowlist secret - public release SHA256
    ),  # pragma: allowlist secret
    (
        "2026-06-05",
        602047785,
        548375182,
        "ababba256966dc8bb005c98fbb4b72faea5301e17dcd949fd5b69e71b8239c50",  # pragma: allowlist secret - public release SHA256
    ),  # pragma: allowlist secret
    (
        "2026-06-06",
        602048286,
        124247983,
        "9546e3fd156d379e2420b8a0ada496ff507cf74aab32b9fc97bd1b6b353fa182",  # pragma: allowlist secret - public release SHA256
    ),  # pragma: allowlist secret
    (
        "2026-06-07",
        602048368,
        139262550,
        "8b6da125b486aea3eb37c752adf6be14d61dd1e6a584f5967bca7077fafe8c3c",  # pragma: allowlist secret - public release SHA256
    ),  # pragma: allowlist secret
]


def verify_archive(path, size, digest):
    if path.stat().st_size != size:
        raise ValueError("official release archive size mismatch")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
            raise ValueError("official release archive checksum mismatch")


def prepare(output, *, execute=False, reuse_archive_dir=None):
    result = {
        "source_id": "COPILOT_AGENT_2026",
        "release_tag": RELEASE_TAG,
        "release_id": RELEASE_ID,
        "archive_bytes": sum(item[2] for item in ARCHIVES),
        "archives": len(ARCHIVES),
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "status": "PLAN_ONLY",
    }
    if not execute:
        return result
    output = Path(output).resolve()
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repository):
        raise ValueError("original assets must remain outside repository")
    if output.exists():
        raise ValueError("fresh output required; preserve and inspect partial work")
    if reuse_archive_dir is not None:
        reuse_archive_dir = Path(reuse_archive_dir).resolve()
        for day, _, size, digest in ARCHIVES:
            verify_archive(reuse_archive_dir / f"date.{day}.tar.gz", size, digest)
    output.mkdir(parents=True, mode=0o700)
    archives, data, staging = output / "archives", output / "data", output / "staging"
    for directory in (archives, data, staging):
        directory.mkdir(mode=0o700)
    result.update(status="PREPARING_ORIGINAL_METADATA_NOT_DELIVERED", days=[], files=[])
    receipt = output / "preparation-public.json"
    try:
        for day, asset_id, size, digest in ARCHIVES:
            name = f"date.{day}.tar.gz"
            archive = archives / name
            partial = archives / (name + ".partial")
            if reuse_archive_dir is not None:
                original = reuse_archive_dir / name
                try:
                    os.link(original, partial)
                except OSError:
                    shutil.copyfile(original, partial)
            else:
                subprocess.run(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--max-time",
                        "1800",
                        "--max-filesize",
                        str(size),
                        "--header",
                        "Accept: application/octet-stream",
                        "--output",
                        str(partial),
                        f"https://api.github.com/repos/Azure/AzurePublicDataset/releases/assets/{asset_id}",
                    ],
                    check=True,
                )
            verify_archive(partial, size, digest)
            partial.rename(archive)
            extracted = staging / day
            verified = extract_verified_archive(
                archive, extracted, expected_sha256=digest, expected_bytes=size
            )
            if verified["qualification"]["date"] != day:
                raise ValueError("release archive day differs from original manifest")
            (extracted / f"date={day}").rename(data / f"date={day}")
            result["days"].append(verified["qualification"])
            result["files"].extend(verified["files"])
            receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["status"] = "ORIGINAL_RELEASE_ASSETS_PREPARED_NOT_PROVIDER_DELIVERY"
    except Exception:
        result["status"] = "FAILED_NOT_COMPLETE"
        raise
    finally:
        receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-data-root", type=Path)
    parser.add_argument("--reuse-archive-dir", type=Path)
    args = parser.parse_args()
    if args.verify_data_root:
        if args.reuse_archive_dir:
            parser.error("read-only source verification does not acquire archives")
        output = args.output.resolve()
        repository = Path(__file__).resolve().parents[1]
        original = args.verify_data_root.resolve()
        if (
            output.exists()
            or output.is_relative_to(repository)
            or output.is_relative_to(original)
        ):
            parser.error("fresh external audit output outside original data required")
        verified = verify_original_directory(
            original, expected_days=[item[0] for item in ARCHIVES]
        )
        result = {
            "source_id": "COPILOT_AGENT_2026",
            "status": "ORIGINAL_DIRECTORY_VERIFIED_NOT_PROVIDER_DELIVERY",
            "release_tag": RELEASE_TAG,
            "source_checkbox_eligible": False,
            **verified,
        }
        output.mkdir(parents=True, mode=0o700)
        (output / "preparation-public.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
        return
    result = prepare(
        args.output, execute=args.execute, reuse_archive_dir=args.reuse_archive_dir
    )
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
