"""Verify all frozen evidence; optionally restore into a separate directory. No devices."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "raw/manifest.json").read_text())
    chunks = []
    for part in manifest["parts"]:
        data = (root / "raw" / part["file"]).read_bytes()
        assert len(data) == part["bytes"]
        assert hashlib.sha256(data).hexdigest() == part["sha256"]
        chunks.append(data)
    archive = b"".join(chunks)
    assert len(archive) == manifest["archive_bytes"]
    assert hashlib.sha256(archive).hexdigest() == manifest["archive_sha256"]
    seen = set()
    with gzip.GzipFile(fileobj=io.BytesIO(archive)) as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as tar:
            for member in tar:
                name = Path(member.name)
                assert (
                    member.isfile()
                    and not name.is_absolute()
                    and ".." not in name.parts
                )
                assert member.name not in seen
                expected = manifest["members"][member.name]
                data = tar.extractfile(member).read()
                assert len(data) == expected["bytes"]
                assert hashlib.sha256(data).hexdigest() == expected["sha256"]
                if args.restore:
                    base = args.restore.resolve()
                    dest = (base / name).resolve()
                    assert dest.is_relative_to(base)
                    if dest.exists():
                        assert dest.read_bytes() == data, (
                            f"Refusing to overwrite: {name}"
                        )
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        dest.write_bytes(data)
                seen.add(member.name)
    assert seen == set(manifest["members"])
    print(
        json.dumps(
            {
                "verified_members": len(seen),
                "archive_sha256": manifest["archive_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
