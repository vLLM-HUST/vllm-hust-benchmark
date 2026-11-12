"""Check all checkpoint bytes and members; optionally extract to a new directory."""

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or ":" in name or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def verify(scratch, destination=None):
    manifest = json.loads((ROOT / "manifest.json").read_bytes())
    scratch.mkdir(parents=True, exist_ok=True)
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    for record in manifest["direct_files"]:
        path = ROOT / safe_name(record["path"])
        assert path.stat().st_size == record["bytes"], path
        assert digest_file(path) == record["sha256"], path
    count = 0
    with tempfile.TemporaryDirectory(prefix="checkpoint-", dir=scratch) as temporary:
        temporary = Path(temporary)
        for capsule in manifest["capsules"]:
            archive_path = temporary / capsule["name"]
            combined = hashlib.sha256()
            with archive_path.open("xb") as output:
                for part in capsule["parts"]:
                    path = ROOT / safe_name(part["path"])
                    block = path.read_bytes()
                    assert len(block) == part["bytes"] <= manifest["part_bytes_max"]
                    assert hashlib.sha256(block).hexdigest() == part["sha256"], path
                    combined.update(block)
                    output.write(block)
            assert archive_path.stat().st_size == capsule["bytes"]
            assert combined.hexdigest() == capsule["sha256"], capsule["name"]
            expected = {record["path"]: record for record in capsule["members"]}
            assert len(expected) == len(capsule["members"])
            seen = set()
            with tarfile.open(archive_path, "r|*") as archive:
                for member in archive:
                    relative = safe_name(member.name)
                    assert member.isfile() and member.name not in seen, member.name
                    record = expected[member.name]
                    assert member.size == record["bytes"]
                    source = archive.extractfile(member)
                    assert source is not None
                    digest = hashlib.sha256()
                    output = None
                    if destination is not None:
                        path = destination / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        output = path.open("xb")
                    try:
                        for block in iter(lambda: source.read(2**20), b""):
                            digest.update(block)
                            if output is not None:
                                output.write(block)
                    finally:
                        if output is not None:
                            output.close()
                    assert digest.hexdigest() == record["sha256"], member.name
                    seen.add(member.name)
                    count += 1
            assert seen == set(expected), capsule["name"]
            archive_path.unlink()
    summary = {
        "all_part_archive_and_member_hashes_passed": True,
        "capsules": len(manifest["capsules"]),
        "members": count,
        "original_host_archives": len(manifest["original_host_archives"]),
        "extracted_to": str(destination) if destination is not None else None,
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    verify(
        args.scratch.resolve(), args.extract_to.resolve() if args.extract_to else None
    )


if __name__ == "__main__":
    main()
