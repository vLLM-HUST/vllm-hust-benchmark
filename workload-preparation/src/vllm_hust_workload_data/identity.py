"""Stable data identities, not research feature extraction."""

from datetime import date, datetime
import hashlib
import json


def portable(value):
    if isinstance(value, bytes):
        return {
            "media_sha256": hashlib.sha256(value).hexdigest(),
            "bytes": len(value),
            "storage": "original_source_only",
        }
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    return value


def canonical(value):
    return json.dumps(
        portable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def key(*parts):
    return hashlib.sha256(canonical(parts).encode("utf-8")).hexdigest()
