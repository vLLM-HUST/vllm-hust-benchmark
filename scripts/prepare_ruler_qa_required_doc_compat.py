"""Opt-in RULER QA control-flow compatibility copy; no task or model execution.

Never edit the original checkout. Questions, required documents, renderer, seeds,
output budgets, client and scorer are not patched.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


PATCH_ID = "ruler-qa-required-doc-floor/v1"
TARGET = "scripts/data/synthetic/qa.py"
ORIGINAL_QA_SHA256 = "f00c0b59cf8698e90831bbf461c2c4f40f2a30f2469534c27455808c717281d0"  # pragma: allowlist secret - pinned publisher source hash
OLD = "used_docs -= incremental"
NEW = "used_docs = max(len(QAS[index + args.pre_samples]['context']), used_docs - incremental)"


def compatibility_copy(source, output, *, allow_required_doc_floor=False):
    if not allow_required_doc_floor:
        raise ValueError("explicit --allow-required-doc-floor opt-in required")
    source, output = Path(source).resolve(), Path(output).resolve()
    if (
        source == output
        or source.is_relative_to(output)
        or output.is_relative_to(source)
    ):
        raise ValueError("runtime copy must be separate from immutable source")
    original_path = source / TARGET
    raw = original_path.read_bytes()
    if (
        original_path.is_symlink()
        or hashlib.sha256(raw).hexdigest() != ORIGINAL_QA_SHA256
    ):
        raise ValueError(
            "pinned publisher QA source mismatch; do not patch unknown code"
        )
    text = raw.decode("utf-8")
    if text.count(OLD) != 1:
        raise ValueError("expected unique original decrement assignment")
    updated = text.replace(OLD, NEW)
    original_tree, patched_tree = ast.parse(text), ast.parse(updated)
    original_renderer = next(
        node
        for node in original_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_input_output"
    )
    patched_renderer = next(
        node
        for node in patched_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_input_output"
    )
    if ast.dump(original_renderer) != ast.dump(patched_renderer):
        raise ValueError("question renderer changed; refuse compatibility copy")
    shutil.copytree(
        source,
        output,
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"),
    )
    output.chmod(0o700)
    patched = output / TARGET
    patched.write_bytes(updated.encode("utf-8"))
    if original_path.read_bytes() != raw:
        raise ValueError("immutable source changed during preparation")
    receipt = {
        "patch_id": PATCH_ID,
        "target_relative_path": TARGET,
        "original_source_sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_source_sha256": hashlib.sha256(patched.read_bytes()).hexdigest(),
        "explicit_opt_in": True,
        "question_renderer_ast_preserved": True,
        "original_checkout_overwritten": False,
        "runtime_copy_created_fresh": True,
        "change_scope": "clamp decrement to original required context document count; no question/document filtering or output-budget change",
        "inference_or_scoring_executed": False,
    }
    (output / "ruler-qa-compatibility.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh, separate runtime copy"
    )
    parser.add_argument("--allow-required-doc-floor", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            compatibility_copy(
                args.source,
                args.output,
                allow_required_doc_floor=args.allow_required_doc_floor,
            )
        )
    )


if __name__ == "__main__":
    main()
