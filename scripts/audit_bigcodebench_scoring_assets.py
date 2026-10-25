"""Offline static preflight; never imports publisher or executes candidate code."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

from run_bigcodebench_original_generation import CODE_REVISION, DATA_SHA, digest

# Observed checksums of the fixed public source, not publisher signatures.
PINS = {
    "bigcodebench/evaluate.py": "6f737ddbec85c8e24d995898bab528b8c28fc5090a488652c8af557ad85f0667",  # pragma: allowlist secret
    "bigcodebench/sanitize.py": "c3b34cd425b0eb62afaf84f4268af787d317ce552685dd6d9b396442c91f47e4",  # pragma: allowlist secret
    "bigcodebench/eval/__init__.py": "d5fd553559ac1b76659ebc32ae30e3e779449ecd31c5201f2301319ceeee01fe",  # pragma: allowlist secret
    "bigcodebench/data/bigcodebench.py": "cac3e3a8425a1fc56816caf6133c193866aadd729c0bb735d8a00a387178007a",  # pragma: allowlist secret
    "Requirements/requirements-eval.txt": "a4d01fb12cbce5223b51f982265cb7975bea770b758cd85cc91b803d3293e39f",  # pragma: allowlist secret
}


def verify_source(root, pins=PINS):
    root = Path(root).resolve(strict=True)
    verified = {}
    for name, expected in pins.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("publisher source path escapes root")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError(f"publisher checksum mismatch: {name}")
        verified[name] = observed
    return verified


def original_imports(row):
    """Reference bodies are indented continuations of complete_prompt."""
    modules = set()
    for text in (row["complete_prompt"] + row["canonical_solution"], row["test"]):
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                modules.update(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError(
                        "relative import requires explicit environment review"
                    )
                modules.add((node.module or "").split(".")[0])
    return sorted(modules)


def audit(root, rows):
    checksums = verify_source(root)
    if len(rows) != 1140 or [r["task_id"] for r in rows] != [
        f"BigCodeBench/{i}" for i in range(1140)
    ]:
        raise ValueError("full original ordered 1140 tasks required")
    return {
        "source_id": "BIGCODEBENCH",
        "status": "STATIC_SCORING_ASSETS_VERIFIED_NOT_EXECUTED",
        "code_revision": CODE_REVISION,
        "publisher_source_checksums": checksums,
        "selected_original_indices": [0, 1, 2],
        "selected_task_static_imports": [
            {"task_id": row["task_id"], "imports": original_imports(row)}
            for row in rows[:3]
        ],
        "static_import_limit": "Not exhaustive: dynamic imports, runtime files and native dependencies still require actual isolated execution and reference checks.",
        "candidate_code_executed": False,
        "official_sanitize_executed": False,
        "official_functional_scoring_executed": False,
        "environment_qualified": False,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-code", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    path = args.data_root / "data/v0.1.4-00000-of-00001.parquet"
    if digest(path) != DATA_SHA:
        raise ValueError("original publisher dataset identity mismatch")
    rows = pq.read_table(path).to_pylist()
    if digest(path) != DATA_SHA:
        raise ValueError("original publisher dataset changed during reading")
    result = audit(args.official_code, rows)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
