"""Bind original Spider dev tasks and complete official suites; execute no SQL.

Private input/reference files remain separate. No prompt invention, predicted
SQL, model invocation, reference substitution or research feature extraction.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

EXPECTED_INPUTS = {
    "dev.json": "30d64a3fccde493226df79687aed9e4a1c0129525baf44f29c0573d914d758a4",  # pragma: allowlist secret - public original file digest
    "tables.json": "61bb20aa401f03164e2d7f3b16509b7b5f79cc9c943ca7bd159046df1159e2ed",  # pragma: allowlist secret - public original file digest
}
SUITE_ARCHIVE_SHA = "9ec24ea8debc6bd04abfe137b5f1a739b5a8836f32c0464e4dfc94eb7f41da96"  # pragma: allowlist secret - observed official archive digest
DEV_ROWS = 1034


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(*, data_root, suite_root, suite_archive, output, count=3):
    data_root, suite_root, suite_archive, output = (
        Path(p).resolve() for p in (data_root, suite_root, suite_archive, output)
    )
    if output.exists() or any(
        output.is_relative_to(p) or p.is_relative_to(output)
        for p in (data_root, suite_root, suite_archive)
    ):
        raise ValueError("fresh private output separate from original assets required")
    for name, expected in EXPECTED_INPUTS.items():
        path = data_root / name
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("original Spider dev/schema identity differs")
    if digest(suite_archive) != SUITE_ARCHIVE_SHA:
        raise ValueError("original complete official suite archive identity differs")
    rows = json.loads((data_root / "dev.json").read_bytes())
    schemas = json.loads((data_root / "tables.json").read_bytes())
    if len(rows) != DEV_ROWS or type(count) is not int or not 1 <= count <= len(rows):
        raise ValueError("fixed positive original dev prefix required")
    by_id = {s["db_id"]: s for s in schemas}
    if len(by_id) != len(schemas):
        raise ValueError("duplicate original schema IDs")
    selected = rows[:count]
    db_ids = sorted({r["db_id"] for r in selected})
    suites = []
    with zipfile.ZipFile(suite_archive) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("duplicate original suite archive paths")
        for db_id in db_ids:
            if not re.fullmatch(r"[A-Za-z0-9_]+", db_id) or db_id not in by_id:
                raise ValueError("invalid or absent original database schema")
            relative_names = [
                n
                for n in names
                if n.startswith(f"database/{db_id}/") and n.endswith(".sqlite")
            ]
            expected_files = {Path(n).name for n in relative_names}
            directory = suite_root / db_id
            actual_files = {p.name for p in directory.glob("*.sqlite")}
            if not expected_files or expected_files != actual_files:
                raise ValueError("complete original per-database suite required")
            if f"{db_id}.sqlite" not in expected_files:
                raise ValueError("official scorer base database absent")
            for name in relative_names:
                path = directory / Path(name).name
                if path.is_symlink() or not path.resolve().is_relative_to(suite_root):
                    raise ValueError("escaped original suite path")
                raw = archive.read(name)
                expected = hashlib.sha256(raw).hexdigest()
                if not raw.startswith(b"SQLite format 3\0") or digest(path) != expected:
                    raise ValueError("original SQLite suite member identity differs")
                suites.append(
                    {"archive_member": name, "path": str(path), "sha256": expected}
                )
    inputs, references, gold = [], [], []
    required = {
        "db_id",
        "query",
        "query_toks",
        "query_toks_no_value",
        "question",
        "question_toks",
        "sql",
    }
    for index, row in enumerate(selected):
        if set(row) != required or any(c in row["query"] for c in "\r\n\t"):
            raise ValueError(
                "original task fields or gold line format differ; do not rewrite"
            )
        inputs.append(
            {
                "original_dev_index": index,
                "db_id": row["db_id"],
                "question": row["question"],
                "schema": by_id[row["db_id"]],
            }
        )
        references.append({"original_dev_index": index, "original_record": row})
        gold.append(row["query"] + "\t" + row["db_id"] + "\n")
    output.mkdir(mode=0o700, parents=True)
    for name, values in (
        ("inputs.private.jsonl", inputs),
        ("references.private.jsonl", references),
    ):
        with (output / name).open("x", encoding="utf-8") as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    (output / "gold.private.txt").write_text(
        "".join(gold), encoding="utf-8", newline="\n"
    )
    report = {
        "source_id": "SPIDER",
        "status": "ORIGINAL_DEV_AND_COMPLETE_SUITE_BOUND_NATIVE_EXECUTION_PENDING",
        "selected_original_dev_indices": list(range(count)),
        "original_dev_rows": len(rows),
        "database_ids": db_ids,
        "suite_files": suites,
        "input_file": "inputs.private.jsonl",
        "reference_file": "references.private.jsonl",
        "reference_sql_in_model_inputs": False,
        "raw_original_fields_preserved": True,
        "sql_statements_executed": 0,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    (output / "preparation-public.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "suite-root", "suite-archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
