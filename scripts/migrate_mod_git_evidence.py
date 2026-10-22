"""Freeze public MOD Git evidence byte-for-byte; never remove source files."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path, PurePosixPath

from migrate_devhub_evidence import SECRET, git, tree

CONTEXT = (
    "README.md",
    "EVIDENCE.md",
    "MOD_METADATA.json",
    "PROVENANCE.json",
    "LICENSE",
    "NOTICE",
)


def invalid_json_paths(payloads):
    invalid = []
    for path, raw in sorted(payloads.items()):
        if path.endswith(".json"):
            try:
                json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                invalid.append(path)
    return invalid


def prepare(
    source: Path,
    repository: str,
    destination: Path,
    prefixes=("evidence/",),
    context=CONTEXT,
) -> dict:
    for value in (*prefixes, *context):
        path = PurePosixPath(value)
        if not value or path.is_absolute() or ".." in path.parts or str(path) == ".":
            raise ValueError("Unsafe source selection")
    if any(not prefix.endswith("/") for prefix in prefixes):
        raise ValueError("Source prefixes must name directories")
    if destination.exists():
        raise ValueError("Refusing to overwrite a custody snapshot")
    metadata = json.loads(subprocess.check_output(["gh", "api", f"repos/{repository}"]))
    if metadata.get("visibility") != "public" or metadata.get("private"):
        raise ValueError("Only explicitly public source repositories are admitted")
    commit = git(source, "rev-parse", "HEAD").decode().strip()
    entries = tree(source, commit)
    selected = {
        p: oid
        for p, oid in entries.items()
        if p.startswith(tuple(prefixes)) or p in context
    }
    if not selected:
        raise ValueError("Empty source scope")
    rows = []
    payloads = {}
    for path, oid in sorted(selected.items()):
        raw = git(source, "cat-file", "blob", oid)
        if SECRET.search(raw):
            raise ValueError(f"Credential review required: {path}; value suppressed")
        payloads[path] = raw
        rows.append(
            {
                "path": path,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_blob": oid,
            }
        )
    for path, raw in payloads.items():
        target = destination / "snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    receipt = {
        "schema": "mod-git-custody/v1",
        "source_repository": repository,
        "source_commit": commit,
        "source_visibility": "public",
        "scope": {"prefixes": list(prefixes), "context": list(context)},
        "admission": "custody-only; original failures and pairing retained; no new performance qualification",
        "files": rows,
        "invalid_json_paths": invalid_json_paths(payloads),
    }
    (destination / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    verify(destination)
    return receipt


def verify(destination: Path) -> None:
    receipt = json.loads((destination / "manifest.json").read_text())
    paths = {row["path"] for row in receipt["files"]}
    if len(paths) != len(receipt["files"]):
        raise ValueError("Duplicate custody path")
    actual = {
        p.relative_to(destination / "snapshot").as_posix()
        for p in (destination / "snapshot").rglob("*")
        if p.is_file()
    }
    if actual != paths:
        raise ValueError("Missing or unindexed custody files")
    for row in receipt["files"]:
        raw = (destination / "snapshot" / row["path"]).read_bytes()
        if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"Custody integrity failure: {row['path']}")
    payloads = {path: (destination / "snapshot" / path).read_bytes() for path in paths}
    if receipt["invalid_json_paths"] != invalid_json_paths(payloads):
        raise ValueError("Invalid JSON inventory mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "verify"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--repository")
    parser.add_argument("--prefix", action="append")
    parser.add_argument("--context", action="append", default=[])
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "verify":
        verify(args.destination)
    else:
        if args.source is None or args.repository is None:
            parser.error("prepare requires --source and --repository")
        receipt = prepare(
            args.source,
            args.repository,
            args.destination,
            tuple(args.prefix or ["evidence/"]),
            (*CONTEXT, *args.context),
        )
        print(
            json.dumps(
                {
                    "repository": args.repository,
                    "files": len(receipt["files"]),
                    "bytes": sum(r["bytes"] for r in receipt["files"]),
                }
            )
        )


if __name__ == "__main__":
    main()
