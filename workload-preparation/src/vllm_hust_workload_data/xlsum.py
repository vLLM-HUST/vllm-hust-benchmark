"""Exact publisher XL-Sum test extraction; no generated or normalized tasks."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

from .assets import asset_path, iter_rows, sha256
from .registry import source_spec


FIELDS = {"id", "url", "title", "summary", "text"}


def validate_rows(rows):
    """Validate the pinned five-field schema without discarding source fields."""
    seen = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or set(row) != FIELDS
            or any(not isinstance(row[k], str) for k in FIELDS)
        ):
            raise ValueError("XL-Sum original five string fields required")
        if any(not row[k].strip() for k in ("id", "summary", "text")):
            raise ValueError("XL-Sum original task, reference and ID must be nonempty")
        if row["id"] in seen:
            raise ValueError("duplicate XL-Sum ID within one original language file")
        seen.add(row["id"])
        yield row


def prepare_original_archives(*, archive_root, output):
    """Verify all registered archives, then copy only exact test member bytes."""
    source = source_spec("XLSUM")
    output = Path(output)
    if output.exists():
        raise ValueError("fresh output directory required")
    archives = {f["path"]: f for f in source["files"] if f["role"] == "source_archive"}
    tests = [f for f in source["files"] if f["role"] == "data"]
    if not tests or len(archives) != len(tests):
        raise ValueError("pinned original archive/test pairs required")
    for name, spec in archives.items():
        path = asset_path(Path(archive_root), name)
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError("XL-Sum original archive identity mismatch")
    # Validate every member before creating the output directory.
    verified = []
    for spec in tests:
        path = asset_path(Path(archive_root), spec["archive_path"])
        with tarfile.open(path, "r:bz2") as archive:
            matches = [m for m in archive if m.name == spec["archive_member"]]
            if len(matches) != 1 or not matches[0].isfile():
                raise ValueError("unique regular publisher test member required")
            with archive.extractfile(matches[0]) as stream:
                raw = stream.read()
        if (
            len(raw) != spec["bytes"]
            or hashlib.sha256(raw).hexdigest() != spec["sha256"]
        ):
            raise ValueError("XL-Sum original test member identity mismatch")
        rows = (json.loads(line) for line in io.StringIO(raw.decode("utf-8-sig")))
        if sum(1 for _ in validate_rows(rows)) != spec["rows"]:
            raise ValueError("XL-Sum original row count mismatch")
        target = (output / spec["path"]).resolve()
        if not target.is_relative_to(output.resolve()):
            raise ValueError("test path escapes output directory")
        verified.append((spec, raw))
    output.mkdir(parents=True, mode=0o700)
    for spec, raw in verified:
        target = output / spec["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return {
        "source_id": "XLSUM",
        "revision": source["revision"],
        "language_variants": len(tests),
        "original_test_rows": sum(f["rows"] for f in tests),
        "output": str(output),
        "native_generation_or_scoring_verified": False,
        "license": source["access"]["license"],
    }


def original_rows(path):
    yield from validate_rows(iter_rows(Path(path)))
