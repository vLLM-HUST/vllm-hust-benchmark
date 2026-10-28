"""Prepare a private pinned ARM64 Node runtime; no system or model changes."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import subprocess
import tarfile

VERSION = "22.14.0"
PREFIX = f"node-v{VERSION}-linux-arm64"
SHA256 = "08bfbf538bad0e8cbb0269f0173cca28d705874a67a22f60b57d99dc99e30050"  # pragma: allowlist secret - official Node release checksum
RELEASE = "https://nodejs.org/en/blog/release/v22.14.0"


def validate_archive(archive):
    if hashlib.sha256(Path(archive).read_bytes()).hexdigest() != SHA256:
        raise ValueError("official Node release checksum differs")
    with tarfile.open(archive, "r:xz") as stream:
        for item in stream.getmembers():
            path = PurePosixPath(item.name)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != PREFIX:
                raise ValueError("contained official Node prefix required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "url": f"https://nodejs.org/dist/v{VERSION}/{PREFIX}.tar.xz",
                    "sha256": SHA256,
                    "official_checksum_source": RELEASE,
                    "model_calls": 0,
                }
            )
        )
        return
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        parser.error("this explicit artifact requires Linux ARM64")
    if args.archive is None or args.output.exists():
        parser.error("existing archive and fresh private output required")
    validate_archive(args.archive)
    args.output.mkdir(mode=0o700, parents=True)
    with tarfile.open(args.archive, "r:xz") as stream:
        stream.extractall(args.output, filter="data")
    root = args.output / PREFIX
    node = root / "bin/node"
    npm_cli = root / "lib/node_modules/npm/bin/npm-cli.js"
    version = subprocess.check_output([str(node), "--version"], text=True).strip()
    if version != f"v{VERSION}":
        raise ValueError("extracted Node version differs")
    npm = subprocess.check_output(
        [str(node), str(npm_cli), "--version"], text=True
    ).strip()
    receipt = {
        "status": "PRIVATE_NODE_RUNTIME_VERIFIED_NOT_SCHEMA_VALIDATION",
        "node": version,
        "npm": npm,
        "archive_sha256": SHA256,
        "checksum_source": RELEASE,
        "system_environment_changed": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    (args.output / "receipt-public.json").write_text(
        json.dumps(receipt, indent=2), "utf-8"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
