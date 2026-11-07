"""Explicitly disclosed indentation-only compatibility copy, never in-place."""

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

ORIGINAL_BYTES = 25214
ORIGINAL_SHA256 = "cc9234a9003792538acc5e01579f484d16446da696319cb0c34a7f3466990ad8"  # pragma: allowlist secret - verified public publisher software digest
BLOCK = (
    b"            except json.JSONDecodeError:\n"
    b'                self.logger.error("Failed to parse JSON from assistant\'s answer.")\n'
    b"                return {}\n"
)


def compatibility_copy(raw):
    if len(raw) != ORIGINAL_BYTES or hashlib.sha256(raw).hexdigest() != ORIGINAL_SHA256:
        raise ValueError("fixed original evaluator identity required; no guessed patch")
    if raw.count(BLOCK) != 1:
        raise ValueError("one exact original malformed exception block required")
    replacement = b"".join(line[4:] for line in BLOCK.splitlines(keepends=True))
    patched = raw.replace(BLOCK, replacement, 1)
    ast.parse(patched)
    before, after = raw.splitlines(keepends=True), patched.splitlines(keepends=True)
    changed = [
        i + 1 for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b
    ]
    if len(changed) != 3 or any(before[i - 1][4:] != after[i - 1] for i in changed):
        raise ValueError(
            "compatibility patch must only remove four leading spaces on three lines"
        )
    return patched, changed


def prepare(source, output, *, execute=False):
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(source.parent)
        or source.is_relative_to(output)
    ):
        raise ValueError(
            "fresh independent output directory required; original source is never modified"
        )
    raw = source.read_bytes()
    patched, changed = compatibility_copy(raw)
    report = {
        "source_id": "MULTIAGENTBENCH",
        "status": "COMPATIBILITY_COPY_CREATED_NOT_NATIVE_EXECUTION"
        if execute
        else "PLAN_ONLY",
        "original_sha256": ORIGINAL_SHA256,
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
        "changed_original_lines": changed,
        "change": "Remove four leading spaces from except and its two handlers; preserve try/except/else and all other bytes",
        "unmodified_official_source_claim": False,
        "original_source_modified": False,
        "author_module_imported": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }
    if execute:
        output.mkdir(mode=0o700)
        (output / "evaluator.py").write_bytes(patched)
        difference = "".join(
            difflib.unified_diff(
                raw.decode().splitlines(True),
                patched.decode().splitlines(True),
                fromfile="official/evaluator.py",
                tofile="compat/evaluator.py",
            )
        )
        (output / "COMPATIBILITY.patch").write_text(difference, encoding="utf-8")
        (output / "compatibility-public.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, execute=args.execute), indent=2))


if __name__ == "__main__":
    main()
