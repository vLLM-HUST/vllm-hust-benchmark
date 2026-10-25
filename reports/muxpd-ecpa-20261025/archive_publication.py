"""Pack an explicitly frozen website-publication inventory without changing bytes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_member(name: str) -> None:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe member: {name}")


def verify(archive: Path, index: dict) -> dict:
    expected = {row["member"]: row for row in index["files"]}
    if len(expected) != len(index["files"]):
        raise ValueError("Duplicate index member")
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            safe_member(member.name)
            if (
                not member.isfile()
                or member.name in seen
                or member.name not in expected
            ):
                raise ValueError(f"Unexpected member: {member.name}")
            source = stream.extractfile(member)
            if source is None:
                raise ValueError("No member content")
            raw = source.read()
            row = expected[member.name]
            if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
                raise ValueError(f"Changed member: {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Missing member")
    raw = archive.read_bytes()
    return {
        "verified": True,
        "files": len(seen),
        "archive": archive.name,
        "bytes": len(raw),
        "sha256": digest(raw),
        "scope": "Every downloaded or local member matches the frozen index; no model execution or performance inference.",
    }


def pack(inventory: Path, output: Path) -> dict:
    inputs = json.loads(inventory.read_text())
    if inputs.get("status") != "terminal":
        raise ValueError("Only a terminal inventory can be archived")
    sources = {}
    for row in inputs["files"]:
        name = row["member"]
        safe_member(name)
        path = Path(row["source"])
        if name in sources or not path.is_absolute() or path.is_symlink():
            raise ValueError(f"Duplicate or unsafe source: {name}")
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
            raise ValueError(f"Frozen source changed: {name}")
        sources[name] = (row, raw)
    if not sources:
        raise ValueError("Empty inventory")
    inventory_member = "provenance/frozen-inventory.json"
    if inventory_member in sources:
        raise ValueError("Inventory uses reserved provenance member")
    inventory_raw = inventory.read_bytes()
    sources[inventory_member] = (
        {
            "member": inventory_member,
            "source": str(inventory.resolve()),
            "bytes": len(inventory_raw),
            "sha256": digest(inventory_raw),
        },
        inventory_raw,
    )
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "muxpd-website-publication-20261025.tar.gz"
    with (
        archive.open("xb") as file,
        gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as stream,
    ):
        for name, (_, raw) in sorted(sources.items()):
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            member.mode = 0o644
            member.mtime = 0
            stream.addfile(member, io.BytesIO(raw))
    index = {
        "schema": "muxpd-website-publication-archive-v1",
        "scope": inputs["scope"],
        "inventory_sha256": digest(inventory_raw),
        "files": [row for row, _ in (sources[name] for name in sorted(sources))],
    }
    index_path = output / "website-publication-archive-index.json"
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    result = verify(archive, index)
    verification = output / "website-publication-archive-verification.json"
    verification.write_text(json.dumps(result, indent=2) + "\n")
    (output / "WEBSITE-PUBLICATION-SHA256SUMS").write_text(
        "".join(
            f"{digest(path.read_bytes())}  {path.name}\n"
            for path in (archive, index_path, verification)
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack_parser = sub.add_parser("pack")
    pack_parser.add_argument("--inventory", type=Path, required=True)
    pack_parser.add_argument("--output", type=Path, required=True)
    check_parser = sub.add_parser("verify")
    check_parser.add_argument("--archive", type=Path, required=True)
    check_parser.add_argument("--index", type=Path, required=True)
    args = parser.parse_args()
    result = (
        pack(args.inventory, args.output)
        if args.command == "pack"
        else verify(args.archive, json.loads(args.index.read_text()))
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
