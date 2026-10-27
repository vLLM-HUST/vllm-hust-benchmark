"""Fetch an immutable benchmark capsule into an external, new directory.

No archive code is executed. All member names, types, lengths and hashes are
checked before writing. A failed fetch never falls back to MOD-local evidence.
"""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(url, limit):
    if not url.startswith(
        (
            "https://raw.githubusercontent.com/vLLM-HUST/vllm-hust-benchmark/",
            "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/",
        )
    ):
        raise ValueError("Only canonical benchmark URLs are allowed")
    with urlopen(
        Request(url, headers={"User-Agent": "benchmark-custody/1.0"}), timeout=90
    ) as response:
        raw = response.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Download exceeds declared size")
    return raw


def verify(manifest, raw):
    archive = manifest["archive"]
    if len(raw) != archive["bytes"] or digest(raw) != archive["sha256"]:
        raise ValueError("Archive checksum/size mismatch")
    expected = {}
    for row in manifest["files"]:
        path = PurePosixPath(row["path"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != row["path"]
            or str(path) == "."
        ):
            raise ValueError("Unsafe member path")
        if row["path"] in expected:
            raise ValueError("Duplicate manifest path")
        expected[row["path"]] = row
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as bundle:
        for member in bundle:
            if member.isdir():
                continue
            prefix = archive["member_prefix"]
            if not member.isfile() or not member.name.startswith(prefix):
                raise ValueError("Unexpected archive member type or prefix")
            name = member.name[len(prefix) :]
            if name not in expected or name in payloads:
                raise ValueError("Unexpected or duplicate archive member")
            row = expected[name]
            if member.size != row["bytes"]:
                raise ValueError("Member size mismatch")
            content = bundle.extractfile(member).read()
            if digest(content) != row["sha256"]:
                raise ValueError("Member checksum mismatch")
            payloads[name] = content
    if payloads.keys() != expected.keys():
        raise ValueError("Missing archive members")
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path)
    source.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "BENCHMARK_EVIDENCE.json",
    )
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    repository = Path(__file__).resolve().parents[1]
    if (
        not args.destination.is_absolute()
        or args.destination.exists()
        or repository == destination
        or repository in destination.parents
    ):
        raise ValueError("Use a new absolute directory outside the source checkout")
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
    else:
        config = json.loads(args.config.read_text())
        raw_manifest = fetch(config["manifest_url"], 2_000_000)
        if digest(raw_manifest) != config["manifest_sha256"]:
            raise ValueError("Manifest checksum mismatch")
        manifest = json.loads(raw_manifest)
    raw = fetch(manifest["archive"]["url"], manifest["archive"]["bytes"])
    payloads = verify(manifest, raw)
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in payloads.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    print(f"Verified {len(payloads)} original files in {destination}")


if __name__ == "__main__":
    main()
