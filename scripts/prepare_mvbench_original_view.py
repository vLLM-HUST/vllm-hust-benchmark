"""Prepare the original MVBench_MP4 table from a complete pinned video snapshot.

Plan by default. No downloads, login, model requests or source modifications.
The table generator is extracted verbatim from the verified author implementation.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

REVISION = "a776e554280b99b70f00cc3eacd69a65e0727efc"  # pragma: allowlist secret - pinned public dataset revision
SOURCE_SHA256 = "1d724826e637a09c4b8445d4af9021c0ae52fabb1bdd9a8e74ae53c0ff848639"  # pragma: allowlist secret - original code digest
TABLE_MD5 = "5c8c6f8b7972c2de65a629590f7c42f5"  # pragma: allowlist secret - author table identity


def original_generator(source):
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Unverified original MVBench implementation")
    module = ast.parse(source)
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "MVBench_MP4"
    )
    prepare = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_dataset"
    )
    functions = [
        node
        for node in ast.walk(prepare)
        if isinstance(node, ast.FunctionDef) and node.name == "generate_tsv"
    ]
    if len(functions) != 1:
        raise ValueError("Original generator is ambiguous")
    return compile(
        ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
        "original-mvbench-generator",
        "exec",
    )


def prepare_view(checkout, snapshot):
    import os

    import pandas as pd

    snapshot = snapshot.resolve(strict=True)
    if snapshot.name != REVISION:
        raise ValueError("Pinned video snapshot directory required")
    source = (checkout / "vlmeval/dataset/mvbench.py").read_bytes()
    generator = original_generator(source)
    raw = json.loads((snapshot / "test.json").read_text(encoding="utf-8"))
    rows = []
    for task, items in raw["meta"].items():
        for item in items:
            media = (snapshot / raw["root"][task] / item["video"]).resolve(strict=True)
            if not media.is_file():
                raise ValueError("Missing original media")
            rows.append((task, item))
    if len(raw["meta"]) != 20 or len(rows) != 4000:
        raise ValueError("Complete original 20-type/4000-task snapshot required")
    table = snapshot / "MVBench_MP4.tsv"
    if table.exists() and hashlib.md5(table.read_bytes()).hexdigest() != TABLE_MD5:
        raise ValueError("Existing unexpected table retained; no overwrite")
    namespace = {
        "os": os,
        "osp": os.path,
        "pd": pd,
        "load": lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        "md5": lambda path: hashlib.md5(Path(path).read_bytes()).hexdigest(),
        "dataset_name": "MVBench_MP4",
        "dataset_path": str(snapshot),
        "self": SimpleNamespace(MP4_MD5=TABLE_MD5),
    }
    exec(generator, namespace)
    namespace["generate_tsv"](str(snapshot))
    actual = hashlib.md5(table.read_bytes()).hexdigest()
    if actual != TABLE_MD5:
        raise ValueError("Generated original table differs from author identity")
    return {
        "status": "ORIGINAL_FULL_TABLE_PREPARED_NOT_NATIVE_EXECUTION",
        "original_tasks": 4000,
        "task_types": 20,
        "table_md5": actual,
        "model_calls": 0,
        "source_checkbox": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        prepare_view(args.official_checkout, args.snapshot)
        if args.execute
        else {
            "status": "PLAN_ONLY",
            "revision": REVISION,
            "original_tasks": 4000,
            "task_types": 20,
            "downloads": False,
            "model_calls": 0,
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
