"""Original Spider split/schema/database relations; no SQL or model execution."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

SPLITS = {
    "train_spider.json": ("tables.json", "database", 7000),
    "train_others.json": ("tables.json", "database", 1659),
    "dev.json": ("tables.json", "database", 1034),
    "test.json": ("test_tables.json", "test_database", 2147),
}
SCHEMAS = {
    "tables.json": "61bb20aa401f03164e2d7f3b16509b7b5f79cc9c943ca7bd159046df1159e2ed",  # pragma: allowlist secret - original public file digest
    "test_tables.json": "7760883d07453737f70aa5ff4c21fb4e14d2a77c06a063a3c023f4905b7faee0",  # pragma: allowlist secret - original public file digest
}


def original_rows(path):
    path = Path(path)
    if path.name not in SPLITS:
        raise ValueError("explicit original Spider split filename required")
    schema_name, namespace, expected_rows = SPLITS[path.name]
    root = path.parent.resolve()
    schema_path = root / schema_name
    if schema_path.is_symlink():
        raise ValueError("original schema must not escape through a symlink")
    schema_raw = schema_path.read_bytes()
    if hashlib.sha256(schema_raw).hexdigest() != SCHEMAS[schema_name]:
        raise ValueError("original split-specific Spider schema identity differs")
    schemas = json.loads(schema_raw)
    by_id = {row["db_id"]: row for row in schemas}
    if len(by_id) != len(schemas):
        raise ValueError("duplicate original schema database IDs")
    rows = json.loads(path.read_bytes())
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError("complete original split row count differs")
    assets = {}
    fields = {
        "db_id",
        "question",
        "question_toks",
        "query",
        "query_toks",
        "query_toks_no_value",
        "sql",
    }
    for row in rows:
        if (
            set(row) != fields
            or not isinstance(row["question"], str)
            or not isinstance(row["query"], str)
        ):
            raise ValueError("seven original Spider fields required, no coercion")
        db_id = row["db_id"]
        if (
            not isinstance(db_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_]+", db_id)
            or db_id not in by_id
        ):
            raise ValueError("original split-specific schema relation missing")
        if db_id not in assets:
            relative = f"{namespace}/{db_id}/{db_id}.sqlite"
            database = root / relative
            if database.is_symlink() or not database.resolve().is_relative_to(root):
                raise ValueError("database asset escaped original root")
            with database.open("rb") as stream:
                if stream.read(16) != b"SQLite format 3\0":
                    raise ValueError("original SQLite asset header differs")
                stream.seek(0)
                checksum = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
                sha = checksum.hexdigest()
            assets[db_id] = {
                "relative_path": relative,
                "namespace": namespace,
                "sha256": sha,
            }
        yield {
            **deepcopy(row),
            "schema": deepcopy(by_id[db_id]),
            "database_asset": deepcopy(assets[db_id]),
            "original_split": path.name,
        }
