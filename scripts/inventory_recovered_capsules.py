"""Inventory calibration/source capsules without executing or deserializing them."""

import argparse
import gzip
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from migrate_devhub_evidence import SECRET

SUFFIXES = {".pt", ".gz", ".tar", ".patch"}
EXCLUDED = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "core",
    "ascend",
    "plugin",
    "plugins",
    "build",
    "dist",
    "wheels",
    "uv",
    "ascend-wheel",
    "site-packages",
    "model",
    "node_modules",
}
LIMIT = 100_000_000


def screen(raw):
    """Bound decompression and scan bytes; never load pickle or extract members."""
    if len(raw) > LIMIT:
        raise ValueError("Capsule exceeds screening limit")
    chunks = [raw]
    if zipfile.is_zipfile(io.BytesIO(raw)):
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            if sum(member.file_size for member in archive.infolist()) > LIMIT:
                raise ValueError("Decompressed capsule exceeds screening limit")
            chunks.extend(archive.read(member) for member in archive.infolist())
    elif raw.startswith(b"\x1f\x8b"):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as archive:
            decoded = archive.read(LIMIT + 1)
        if len(decoded) > LIMIT:
            raise ValueError("Decompressed capsule exceeds screening limit")
        chunks.append(decoded)
    if any(SECRET.search(chunk) for chunk in chunks):
        raise ValueError("Credential-pattern finding; values suppressed")


def inventory(source, output):
    if output.exists():
        raise ValueError("Refusing to overwrite an inventory")
    source = source.resolve()
    selected = []
    excluded = []
    for root in sorted(source.iterdir()):
        if root.is_symlink() or not root.is_dir():
            continue
        if not (root.name.startswith("phase") or root.name in {"runs", "receipts"}):
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            keep = []
            for name in sorted(dirs):
                path = Path(directory) / name
                if (
                    name in EXCLUDED
                    or name.startswith("tiering-storage")
                    or path.is_symlink()
                ):
                    excluded.append(str(path.relative_to(source)))
                else:
                    keep.append(name)
            dirs[:] = keep
            for name in sorted(files):
                path = Path(directory) / name
                if path.is_symlink() or path.suffix not in SUFFIXES:
                    continue
                raw = path.read_bytes()
                screen(raw)
                selected.append(
                    {
                        "original_path": str(path.relative_to(source)),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw),
                    }
                )
    payload = {
        "source_root": str(source),
        "selection": "Calibration and source capsules under phase*/runs/receipts.",
        "included_suffixes": sorted(SUFFIXES),
        "excluded_directories": excluded,
        "limitations": "Not a complete filesystem backup; runtime trees, models and KV storage excluded.",
        "screening": "Bounded raw/ZIP/gzip byte-pattern scan; no pickle execution; not a universal secret scan.",
        "files": selected,
    }
    with output.open("x") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "paths": len(selected),
                "unique_files": len({row["sha256"] for row in selected}),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inventory(args.source, args.output)
