"""Validate raw archives without deserializing or executing any archived content."""

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def inspect_archive(path, metadata_dir=None):
    seen = set()
    expected = None
    manifest = None
    omissions = None
    verified = 0
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = str(PurePosixPath(member.name))
            assert (
                not PurePosixPath(name).is_absolute()
                and ".." not in PurePosixPath(name).parts
            )
            assert name not in seen and member.isfile(), name
            seen.add(name)
            with archive.extractfile(member) as f:
                if name in {"manifest.json", "omissions.json"}:
                    data = f.read()
                    obj = json.loads(data)
                    if name == "manifest.json":
                        manifest = obj
                        if obj["schema"] == "pegaflow-original-text-evidence/v1":
                            expected = {
                                "files/" + r["path"]: (r["bytes"], r["sha256"])
                                for r in obj["files"]
                            }
                        else:
                            assert obj["schema"] == "pegaflow-numerical-evidence/v1"
                            expected = {
                                r["object"]: (r["bytes"], r["sha256"])
                                for r in obj["files"]
                            }
                            assert len(expected) == obj["object_count"]
                        assert len(obj["files"]) == obj["file_count"]
                    else:
                        omissions = obj
                    if metadata_dir:
                        target = metadata_dir / (
                            path.name.removesuffix(".tar.gz") + "-" + name + ".gz"
                        )
                        with gzip.open(target, "wb", compresslevel=6) as out:
                            out.write(data)
                    continue
                assert expected is not None and name in expected, name
                digest = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                assert (size, digest.hexdigest()) == expected[name], name
                verified += 1
    assert expected is not None
    assert set(expected) == seen - {"manifest.json", "omissions.json"}
    # Force gzip CRC/EOF validation even if tar ended before the final gzip trailer.
    with gzip.open(path, "rb") as f:
        for _ in iter(lambda: f.read(1024 * 1024), b""):
            pass
    with path.open("rb") as f:
        h = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
        digest = h.hexdigest()
    result = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "verified_members": verified,
        "logical_paths": manifest["file_count"],
        "logical_bytes": manifest["logical_bytes"],
        "schema": manifest["schema"],
    }
    if omissions:
        result["restricted_files"] = [
            r
            for r in omissions["files"]
            if r["reason"] == "restricted-credential-pattern"
        ]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--metadata-dir", type=Path)
    args = p.parse_args()
    results = []
    if args.metadata_dir:
        args.metadata_dir.mkdir(exist_ok=True, parents=True)
    for path in args.paths:
        result = inspect_archive(path, args.metadata_dir)
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.metadata_dir:
        (args.metadata_dir / "archive-verification.json").write_text(
            json.dumps({"archives": results}, indent=2), encoding="utf-8"
        )
