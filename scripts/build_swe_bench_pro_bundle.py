#!/usr/bin/env python3
"""Package only public runtime files from an operator-verified source checkout."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

from bootstrap_swe_bench_pro import RUNTIME_SOURCE_FILES


def build(root, revision, output):
    files = {name: (root / name).read_bytes() for name in RUNTIME_SOURCE_FILES}
    manifest = {
        "commit": revision,
        "files": {
            name: hashlib.sha256(value).hexdigest() for name, value in files.items()
        },
    }
    files["source-bundle.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with tarfile.open(output, "w:gz") as archive:
        for name, value in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(value)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(value))
    return hashlib.sha256(output.read_bytes()).hexdigest()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    print(build(Path(__file__).resolve().parents[1], args.revision, args.output))
