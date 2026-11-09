"""Verify a calibration capsule and optionally restore it into a new directory."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath


def restore(destination=None):
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    archive = manifest["archive"]
    chunks = []
    for part in archive["parts"]:
        if Path(part["path"]).name != part["path"]:
            raise ValueError("Unsafe part path")
        raw = (root / part["path"]).read_bytes()
        if (
            len(raw) != part["bytes"]
            or hashlib.sha256(raw).hexdigest() != part["sha256"]
        ):
            raise ValueError("Part differs: " + part["path"])
        chunks.append(raw)
    payload = b"".join(chunks)
    if (
        len(payload) != archive["bytes"]
        or hashlib.sha256(payload).hexdigest() != archive["sha256"]
    ):
        raise ValueError("Archive differs from manifest")
    expected = {row["path"]: row for row in manifest["members"]}
    if len(expected) != len(manifest["members"]):
        raise ValueError("Duplicate manifest member")
    if destination is not None:
        destination = destination.resolve()
        destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
        for member in bundle:
            name = member.name
            path = PurePosixPath(name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or ":" in name
            ):
                raise ValueError("Unsafe member: " + name)
            if name not in expected or name in seen:
                raise ValueError("Unexpected member: " + name)
            seen.add(name)
            source = bundle.extractfile(member)
            if source is None:
                raise ValueError("Unreadable member: " + name)
            raw = source.read()
            row = expected[name]
            if (
                len(raw) != row["bytes"]
                or hashlib.sha256(raw).hexdigest() != row["sha256"]
            ):
                raise ValueError("Member differs: " + name)
            if destination is not None:
                output = destination.joinpath(*path.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(raw)
    if seen != set(expected):
        raise ValueError("Archive is missing members")
    print(
        json.dumps(
            {
                "verified_members": len(seen),
                "archive_bytes": len(payload),
                "destination": str(destination) if destination else None,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if (args.destination is None) == (not args.verify_only):
        parser.error("Choose --destination or --verify-only")
    restore(args.destination)
