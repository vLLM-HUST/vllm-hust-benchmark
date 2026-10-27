"""Explicit, contained, content-addressed local asset access."""

from __future__ import annotations

import csv
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path, PureWindowsPath
from typing import Iterator


def asset_path(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError("asset path must be relative to the explicit data root")
    base = Path(root).resolve()
    path = (base / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(base):
        raise ValueError("asset path escapes the explicit data root")
    if not path.is_file():
        raise FileNotFoundError(
            f"asset missing: {relative}; supply data root; no download or fallback"
        )
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(root: Path, spec: dict) -> dict:
    path = asset_path(root, spec["path"])
    actual_hash = sha256(path)
    if not spec.get("sha256"):
        raise ValueError(
            "unfrozen file identity; register a pinned hash before adaptation"
        )
    if actual_hash != spec["sha256"] or (
        spec.get("bytes") is not None and path.stat().st_size != spec["bytes"]
    ):
        raise ValueError(f"asset identity mismatch: {spec['path']}")
    return {"path": spec["path"], "sha256": actual_hash, "bytes": path.stat().st_size}


def _json(text: str):
    def reject(_):
        raise ValueError("non-finite JSON")

    return json.loads(text, parse_constant=reject)


def iter_rows(
    path: Path, *, records_path=(), row_format=None, root_fields=()
) -> Iterator[dict]:
    """Parse data only. Values are never included in parser error messages."""
    try:
        if (
            not isinstance(root_fields, (list, tuple))
            or any(not isinstance(key, str) or not key for key in root_fields)
            or len(set(root_fields)) != len(root_fields)
        ):
            raise ValueError("root fields must be distinct nonempty field names")
        if root_fields and (
            path.suffix != ".json" or row_format is not None or not records_path
        ):
            raise ValueError(
                "root fields require an explicit nested JSON array contract"
            )
        if row_format is not None and row_format != "jsonl":
            raise ValueError("unsupported explicit row format")
        if row_format == "jsonl" and records_path:
            raise ValueError("JSONL cannot have an array records path")
        if row_format == "jsonl" or path.name.endswith((".jsonl", ".jsonl.gz")):
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8-sig") as stream:
                for line in stream:
                    if not line.strip():
                        raise ValueError("blank record")
                    row = _json(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected object record")
                    yield row
        elif path.suffix == ".json":
            rows = _json(path.read_text(encoding="utf-8-sig"))
            if root_fields and (
                not isinstance(rows, dict)
                or any(key not in rows for key in root_fields)
            ):
                raise ValueError("declared original root field missing")
            context = {key: rows[key] for key in root_fields}
            for component in records_path:
                if not isinstance(rows, dict) or component not in rows:
                    raise ValueError("declared JSON records path is missing")
                rows = rows[component]
            if not isinstance(rows, list):
                raise ValueError("expected top-level record array")
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("expected object record")
                if set(row) & set(context):
                    raise ValueError("original root and instance fields collide")
                yield {**deepcopy(context), **row}
        elif path.name.endswith((".csv", ".csv.gz")):
            opener = gzip.open if path.name.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or len(set(reader.fieldnames)) != len(
                    reader.fieldnames
                ):
                    raise ValueError("missing or duplicate CSV headers")
                for row in reader:
                    if None in row or None in row.values():
                        raise ValueError("CSV width mismatch")
                    yield dict(row)
        elif path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise RuntimeError("Parquet requires the optional data extra") from None
            for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
                yield from batch.to_pylist()
        else:
            raise ValueError("unsupported data format")
    except (ValueError, UnicodeError) as error:
        raise ValueError(
            f"invalid data in {path.name} ({type(error).__name__}); payload suppressed"
        ) from None
