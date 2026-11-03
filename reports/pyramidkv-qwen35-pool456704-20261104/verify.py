"""CPU-only public capsule integrity and independent metric recomputation."""

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
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        if sha(ROOT / name) != expected:
            raise ValueError("Outer checksum mismatch: " + name)
    members = json.loads(gzip.decompress((ROOT / "members.json.gz").read_bytes()))
    with tempfile.TemporaryDirectory(prefix="pyramidkv-pressure-verify-") as directory:
        output = Path(directory)
        seen = set()
        for path in sorted(ROOT.glob("raw-*.tar.gz")):
            with tarfile.open(path) as archive:
                names = archive.getnames()
                expected = {
                    name
                    for name, identity in members.items()
                    if identity["archive"] == path.name
                }
                if (
                    len(names) != len(set(names))
                    or set(names) != expected
                    or seen.intersection(names)
                ):
                    raise ValueError("Archive inventory mismatch: " + path.name)
                if not all(member.isfile() for member in archive.getmembers()):
                    raise ValueError("Non-file archive member")
                archive.extractall(output, filter="data")
                seen.update(names)
        if seen != set(members):
            raise ValueError("Incomplete archive set")
        for name, identity in members.items():
            path = output / name
            if (
                path.stat().st_size != identity["bytes"]
                or sha(path) != identity["sha256"]
            ):
                raise ValueError("Member checksum mismatch: " + name)
        original_layout = (output / "native-layout.json").read_bytes()
        subprocess.run(
            [sys.executable, str(output / "verify_native_layout.py")],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if (output / "native-layout.json").read_bytes() != original_layout:
            raise ValueError("Native layout differs from retained confirmation")
        subprocess.run(
            [sys.executable, str(output / "analyze_v2.py")],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if (output / "analysis.json").read_bytes() != (
            ROOT / "analysis.json"
        ).read_bytes():
            raise ValueError("Recomputed analysis differs from publication")
        subprocess.run(
            [sys.executable, str(output / "verify_pressure_coverage.py")],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        if (output / "pressure-coverage.json").read_bytes() != (
            ROOT / "pressure-coverage.json"
        ).read_bytes():
            raise ValueError("Recomputed coverage differs from publication")
    print(
        json.dumps(
            {
                "verified": True,
                "frozen_full_compression_acceptance": json.loads(
                    (ROOT / "analysis.json").read_text()
                )["frozen_full_compression_acceptance"],
                "members": len(members),
                "measured_requests": 352,
                "readiness_requests": 4,
            }
        )
    )


if __name__ == "__main__":
    main()
