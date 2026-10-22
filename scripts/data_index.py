"""Offline navigation over tracked paths; never read or rewrite evidence payloads."""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "docs/data-index.json"
EXCLUDED = {OUTPUT, "DATA_INDEX.md"}
ROUTES = {
    "archive": ("historical-evidence", "preserve-original-append-correction"),
    "submissions": ("run-evidence", "append-new-run"),
    "reports": ("reports-and-evidence", "preserve-raw-version-analysis"),
    "leaderboard-data": ("admission-and-snapshots", "validate-and-regenerate"),
    "website-publication": ("publication-source", "update-source-validate-publish"),
    "docs": ("contracts-and-guides", "edit-contract-with-tests"),
    "schemas": ("schemas", "edit-schema-with-tests"),
    "src/vllm_hust_benchmark/data": ("packaged-contracts", "edit-contract-with-tests"),
    "tests/fixtures": ("synthetic-test-fixtures", "test-only-not-measurement"),
    "scripts": ("helpers", "edit-code-with-tests"),
}
ENTRY_NAMES = {
    "readme.md",
    "manifest.json",
    "release.json",
    "index.json",
    "summary.json",
    "report.json",
    "receipt.json",
    "sha256sums",
    "checksums.sha256",
    "leaderboard_manifest.json",
    "source-audit.json",
    "consolidation.md",
}


def classify(path):
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe repository path: {path}")
    if path in EXCLUDED:
        return None
    for prefix in sorted(ROUTES, key=len, reverse=True):
        if not path.startswith(prefix + "/"):
            continue
        rest = path[len(prefix) + 1 :].split("/")
        collection = prefix + "/" + rest[0] if len(rest) > 1 else prefix
        if prefix == "archive" and rest[0].startswith("mod-git-") and len(rest) > 2:
            collection += "/" + rest[1]
        if (
            prefix == "reports"
            and rest[0] == "vllm-hust-midterm-20261019"
            and len(rest) > 2
        ):
            collection += "/" + rest[1]
        category, policy = ROUTES[prefix]
        return {
            "path": path,
            "collection": collection,
            "category": category,
            "write_policy": policy,
        }
    return None


def tracked_paths(root):
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return sorted(set(raw.decode("utf-8").split("\0")) - {""})


def records(paths):
    return [r for p in sorted(set(paths)) if (r := classify(p)) is not None]


def build(paths):
    groups = {}
    for record in records(paths):
        group = groups.setdefault(
            record["collection"],
            {
                "path": record["collection"],
                "category": record["category"],
                "write_policy": record["write_policy"],
                "files": [],
            },
        )
        group["files"].append(record["path"])
    collections = []
    for group in groups.values():
        files = group.pop("files")
        entries = [p for p in files if PurePosixPath(p).name.lower() in ENTRY_NAMES]
        entries.sort(
            key=lambda p: (
                len(PurePosixPath(p).parts),
                PurePosixPath(p).name.lower() != "readme.md",
                p,
            )
        )
        collections.append(
            {
                **group,
                "file_count": len(files),
                "formats": dict(
                    sorted(
                        Counter(
                            PurePosixPath(p).suffix.lower() or "[no extension]"
                            for p in files
                        ).items()
                    )
                ),
                "entrypoint_count": len(entries),
                "entrypoints": entries[:12],
                "entrypoints_truncated": len(entries) > 12,
            }
        )
    return {
        "schema": "benchmark-data-navigation/v1",
        "generated_by": "python scripts/data_index.py --write",
        "scope": "Tracked repository paths only; includes staged additions. No payload parsing, archive extraction, network access or integrity verification.",
        "limits": "Navigation is not an acceptance verdict. Release members and reconstructed publication bytes are indexed by their own manifests, not expanded here. Counts include docs/helpers and may include repeated copies.",
        "roots": list(ROUTES),
        "file_count": sum(g["file_count"] for g in collections),
        "collections": sorted(collections, key=lambda g: g["path"]),
    }


def render(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write", action="store_true", help="Regenerate only docs/data-index.json"
    )
    mode.add_argument(
        "--check", action="store_true", help="Fail if the committed navigation is stale"
    )
    parser.add_argument(
        "--query",
        default="",
        help="Case-insensitive AND terms matched against paths, not payloads",
    )
    parser.add_argument(
        "--files",
        action="store_true",
        help="List matching tracked files rather than collections",
    )
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error("--limit must be positive")
    if (args.write or args.check) and (args.query or args.files):
        parser.error("--write/--check cannot be combined with query options")
    paths = tracked_paths(ROOT)
    catalog = build(paths)
    output = ROOT / OUTPUT
    if args.write:
        output.write_text(render(catalog), encoding="utf-8")
        print(f"Updated {OUTPUT}: {catalog['file_count']} tracked paths")
        return 0
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != render(
            catalog
        ):
            print(
                "Data index is stale. Stage intended additions/deletions, then run python scripts/data_index.py --write"
            )
            return 1
        print("Data index is current (navigation only, not evidence validation)")
        return 0
    terms = args.query.casefold().split()
    matched = [
        r for r in records(paths) if all(t in r["path"].casefold() for t in terms)
    ]
    if args.files:
        result = matched
    else:
        selected = {r["collection"] for r in matched}
        result = [g for g in catalog["collections"] if g["path"] in selected]
    print(
        render(
            {
                "matched": len(result),
                "truncated": len(result) > args.limit,
                "results": result[: args.limit],
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
