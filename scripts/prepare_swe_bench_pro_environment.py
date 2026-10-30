#!/usr/bin/env python3
"""Fetch pinned upstream sources; never install into or restart a model runtime."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request

from vllm_hust_benchmark.swe_bench_pro_inputs import LOCK_PATH, digest, read


def extract_source(data, output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive.getmembers():
            relative = Path(*Path(member.name).parts[1:])
            target = (output / relative).resolve()
            if target != output and output not in target.parents:
                raise ValueError("Archive path escapes source directory")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                target.chmod(member.mode & 0o777)
            elif member.issym() or member.islnk():
                raise ValueError("Source archive links require explicit review")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    lock = read(LOCK_PATH)
    receipt = {}
    for key, name in [("agent", "agent"), ("official_v1", "official-v1")]:
        pin = lock[key]
        url = f"https://codeload.github.com/{pin['repository']}/tar.gz/{pin['commit']}"
        request = urllib.request.Request(
            url, headers={"User-Agent": "swe-bench-pro-bootstrap/1.0"}
        )
        try:
            data = urllib.request.urlopen(request, timeout=20).read()
        except OSError:
            curl = shutil.which("curl")
            if not curl:
                raise
            data = subprocess.check_output(
                [
                    curl,
                    "--fail",
                    "--location",
                    "--retry",
                    "2",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "10",
                    "--max-time",
                    "120",
                    url,
                ],
                timeout=400,
            )
        destination = args.root / name
        extract_source(data, destination)
        expected = (
            lock["agent_source_files"]
            if key == "agent"
            else {
                **pin["sha256"],
                **pin["nodebb_task_source_sha256"],
                **pin.get("grading_dockerfiles_sha256", {}),
            }
        )
        for relative, checksum in expected.items():
            if digest(destination / relative) != checksum:
                raise ValueError("Upstream source SHA mismatch: " + relative)
        receipt[key] = {
            "commit": pin["commit"],
            "source_path": str(destination.resolve()),
            "archive_sha256": hashlib.sha256(data).hexdigest(),
            "verified_files": len(expected),
        }
    (args.root / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
