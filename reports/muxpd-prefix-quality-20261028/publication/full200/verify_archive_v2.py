"""Verify every regular archive member without extracting/executing contents."""

import argparse
import hashlib
import json
import stat
import tarfile
from pathlib import Path, PurePosixPath


def sha_stream(stream):
    digest = hashlib.sha256()
    total = 0
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def sha_file(path):
    with path.open("rb") as stream:
        return sha_stream(stream)[0]


def safe_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or any(x in ("", ".", "..") for x in name.split("/"))
        or "\\" in name
        or any(ord(c) < 32 for c in name)
        or any(
            x in ("compiler-cache", "__pycache__", ".pytest_cache", ".ruff_cache")
            for x in path.parts
        )
    ):
        raise ValueError(f"Unsafe or excluded archive path: {name!r}")
    return name


def verify(archive, manifest_path, compare_sources=False):
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {safe_name(row["member"]): row for row in manifest["members"]}
    if len(expected) != len(manifest["members"]):
        raise ValueError("Manifest has duplicate members")
    embedded = safe_name(manifest["embedded_manifest_member"])
    if embedded in expected:
        raise ValueError("Embedded manifest cannot list itself")
    seen, rows = set(), []
    allowed = [Path(p).resolve() for p in manifest["allowed_source_roots"]]
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            name = safe_name(member.name)
            if name in seen or not member.isfile() or member.issym() or member.islnk():
                raise ValueError("Duplicate or non-regular member")
            seen.add(name)
            payload = stream.extractfile(member)
            if payload is None:
                raise ValueError("Missing member contents")
            if name == embedded:
                if payload.read() != manifest_bytes:
                    raise ValueError("Embedded manifest bytes differ from sidecar")
                continue
            if name not in expected:
                raise ValueError(f"Unexpected member: {name}")
            wanted = expected[name]
            digest, size = sha_stream(payload)
            if (
                digest != wanted["sha256"]
                or size != wanted["bytes"]
                or size != member.size
            ):
                raise ValueError(f"Member contents differ: {name}")
            source_verified = None
            if compare_sources:
                source = Path(wanted["source"])
                resolved = source.resolve(strict=True)
                if (
                    not any(resolved.is_relative_to(root) for root in allowed)
                    or not stat.S_ISREG(source.lstat().st_mode)
                    or source.is_symlink()
                ):
                    raise ValueError(
                        "Source outside explicitly allowed scope or not regular"
                    )
                if source.stat().st_size != size or sha_file(source) != digest:
                    raise ValueError(f"Original source changed: {source}")
                source_verified = True
            rows.append(
                {
                    "member": name,
                    "bytes": size,
                    "sha256": digest,
                    "source_bytes_verified": source_verified,
                }
            )
    if seen != set(expected) | {embedded}:
        raise ValueError("Archive is missing exact manifest members")
    return {
        "schema": "muxpd-prefix-full-publication-verification-v2",
        "passed": True,
        "scope": "Local archive bytes only; no download, extraction, code execution or new experiment.",
        "archive": {
            "name": archive.name,
            "bytes": archive.stat().st_size,
            "sha256": sha_file(archive),
        },
        "manifest": {"name": manifest_path.name, "sha256": sha_file(manifest_path)},
        "original_member_count": len(expected),
        "archive_member_count_including_embedded_manifest": len(seen),
        "embedded_manifest_exact": True,
        "all_paths_safe": True,
        "all_members_regular_files": True,
        "all_original_members_verified": True,
        "source_comparison_requested": compare_sources,
        "all_original_source_bytes_verified": True if compare_sources else None,
        "members": rows,
        "remote_available_claimed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--compare-sources",
        action="store_true",
        help="Local producer machine only; read explicit manifest source files",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.archive, args.manifest, args.compare_sources)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "members"}, indent=2))


if __name__ == "__main__":
    main()
