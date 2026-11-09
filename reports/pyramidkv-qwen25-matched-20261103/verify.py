"""CPU-only verification of the public diagnostic reproduction capsule."""

import gzip
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha(ROOT / name) != expected:
            raise ValueError("Outer checksum mismatch: " + name)
    members = json.loads(gzip.decompress((ROOT / "members.json.gz").read_bytes()))
    with tempfile.TemporaryDirectory(prefix="pyramidkv-qwen25-verify-") as directory:
        output = Path(directory)
        with tarfile.open(ROOT / "raw.tar.gz") as archive:
            names = archive.getnames()
            if len(names) != len(set(names)) or set(names) != set(members):
                raise ValueError("Archive inventory mismatch")
            archive.extractall(output, filter="data")
        for name, identity in members.items():
            path = output / name
            if (
                path.stat().st_size != identity["bytes"]
                or sha(path) != identity["sha256"]
            ):
                raise ValueError("Member checksum mismatch: " + name)
        subprocess.run(
            [sys.executable, str(output / "analyze.py")],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if (output / "analysis.json").read_bytes() != (
            ROOT / "analysis.json"
        ).read_bytes():
            raise ValueError("Recomputed analysis differs from publication")
    print(
        json.dumps(
            {"verified": True, "members": len(members), "measured_requests": 480}
        )
    )


if __name__ == "__main__":
    main()
