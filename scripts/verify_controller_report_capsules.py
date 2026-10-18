"""Verify public evidence hashes and syntax without rewriting frozen measurements."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root, relative):
    candidate = root / relative
    if (
        Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or candidate.is_symlink()
    ):
        raise ValueError(f"Unsafe evidence path: {relative}")
    if (
        not candidate.resolve().is_relative_to(root.resolve())
        or not candidate.is_file()
    ):
        raise ValueError(f"Missing evidence file: {relative}")
    return candidate


def verify(root):
    root = Path(root).resolve()
    immutable = json.loads((root / "immutable-files.json").read_text())
    for relative, expected in immutable["files"].items():
        if digest(safe_path(root, relative)) != expected:
            raise ValueError(f"Frozen evidence changed: {relative}")
    entries = 0
    for manifest in sorted(root.rglob("SHA256SUMS")):
        for line in manifest.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            if digest(safe_path(manifest.parent, relative)) != expected:
                raise ValueError(f"Checksum mismatch: {manifest.parent / relative}")
            entries += 1
    sources = list(root.rglob("*.py"))
    for source in sources:
        ast.parse(source.read_bytes(), filename=str(source))
    verifiers = [p for p in root.rglob("verify.py") if "sources" not in p.parts]
    for verifier in sorted(verifiers):
        subprocess.run([sys.executable, str(verifier)], check=True, cwd=verifier.parent)
    return {
        "checksum_entries": entries,
        "immutable_files": len(immutable["files"]),
        "python_sources": len(sources),
        "public_verifiers": len(verifiers),
        "status": "PASS",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root)))
