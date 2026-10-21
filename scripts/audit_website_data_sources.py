"""Inventory committed website evidence against committed benchmark objects.

This is a provenance inventory, not a claim that unmatched data is missing.
Uncommitted copies are deliberately excluded from source discovery.
"""

import argparse
import collections
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("data/", "docs/evidence/", "reports/", "op01-evidence/")


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def tree(root, revision):
    entries = {}
    for entry in git(root, "ls-tree", "-rz", revision).split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        if kind == b"blob":
            entries[path.decode()] = oid.decode()
    return entries


def inventory(website, benchmark_revision="HEAD", website_revision="HEAD"):
    sources = collections.defaultdict(list)
    for path, oid in tree(ROOT, benchmark_revision).items():
        sources[oid].append(path)
    rows = []
    for path, oid in tree(website, website_revision).items():
        if not path.startswith(PREFIXES):
            continue
        matches = sorted(sources.get(oid, []))
        rows.append(
            {
                "website_path": path,
                "website_git_blob": oid,
                "status": "existing-identical-source"
                if matches
                else "needs-provenance-review",
                "benchmark_paths": matches,
                "action": "reuse-existing-source"
                if matches
                else "trace-or-migrate-original",
            }
        )
    return {
        "schema": "website-source-consolidation-audit/v1",
        "website_commit": git(website, "rev-parse", f"{website_revision}^{{commit}}")
        .decode()
        .strip(),
        "benchmark_commit": git(ROOT, "rev-parse", f"{benchmark_revision}^{{commit}}")
        .decode()
        .strip(),
        "scope": list(PREFIXES),
        "counts": dict(collections.Counter(row["status"] for row in rows)),
        "limitations": [
            "Non-identical files may be generated projections, reformatted JSON, or archive members.",
            "A matching Git blob proves identical bytes, not experiment comparability or acceptance.",
            "Uncommitted website-publication copies are not accepted as established sources.",
            "Remote bare-metal archives have not been inspected by this audit.",
        ],
        "migration_leads": [
            {
                "path": "/data/codex-build-artifacts/frontier-mods-qwen35-20260925",
                "evidence": "vllm-hust-dev-hub/docs/frontier-mod-campaign-20260925.md",
                "status": "documented-location-not-live-verified",
            },
            {
                "path": "/data/codex-build-work/frontier-mods-qwen35-20260925",  # pragma: allowlist secret
                "evidence": "vllm-hust-dev-hub/config/frontier-mod-campaign-20260925.json",
                "status": "documented-location-not-live-verified",
            },
            {
                "host": "192.168.0.84",
                "path": "/data/users/admin/frontier-mods-qwen35-20260925",
                "status": "user-supplied-location-not-live-verified",
            },
        ],
        "files": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--website", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--benchmark-revision",
        default="HEAD",
        help="Committed benchmark tree to inventory (set to the pre-import revision)",
    )
    parser.add_argument(
        "--website-revision",
        default="HEAD",
        help="Committed website tree to inventory",
    )
    args = parser.parse_args()
    result = inventory(
        args.website.resolve(), args.benchmark_revision, args.website_revision
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result["counts"]))
