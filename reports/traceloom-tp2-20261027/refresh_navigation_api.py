"""Run the repository navigation refresher with verified GitHub Git-object reads.

Used when Git smart-HTTP is unavailable. The existing generator still owns all
snapshot/index semantics; only its read-only website Git transport is adapted.
"""

import argparse
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def api(endpoint):
    return json.loads(subprocess.check_output(["gh", "api", endpoint]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    a = p.parse_args()
    root = a.repo.resolve()
    sys.path.insert(0, str(root / "scripts"))
    import data_entrypoints as generator

    old = json.loads((root / generator.SOURCES).read_text())
    revision = old["website_commit"]
    repository = old["website_repository"]
    commit = api(f"repos/{repository}/commits/{revision}")["sha"]
    if commit != revision:
        raise ValueError("Website revision is not immutable")
    tree = api(f"repos/{repository}/git/trees/{revision}?recursive=1")
    if tree.get("truncated"):
        raise ValueError("Incomplete website tree")
    blobs = {r["path"]: r["sha"] for r in tree["tree"] if r["type"] == "blob"}
    needed = [
        p
        for p in blobs
        if (
            p.endswith((".html", ".js", ".mjs"))
            and ("/" not in p or p.startswith("assets/"))
        )
        or p == "data/benchmark-source-manifest.json"
    ]
    content = {}

    def fetch(path):
        blob = api(f"repos/{repository}/git/blobs/{blobs[path]}")
        if blob["encoding"] != "base64":
            raise ValueError("Unexpected blob encoding")
        raw = base64.b64decode(blob["content"])
        actual = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if actual != blobs[path]:
            raise ValueError(("Git blob identity mismatch", path))
        return path, raw

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        content.update(pool.map(fetch, needed))

    def readonly_git(repo, *args):
        if args == ("rev-parse", "--verify", f"{revision}^{{commit}}"):
            return (revision + "\n").encode()
        if args == ("ls-tree", "-rz", "--name-only", revision):
            return ("\0".join(sorted(blobs)) + "\0").encode()
        if len(args) == 2 and args[0] == "show" and args[1].startswith(revision + ":"):
            return content[args[1].split(":", 1)[1]]
        raise ValueError(("Unexpected Git request", args))

    generator.git = readonly_git
    raise SystemExit(
        generator.main(
            ["--refresh", "--website", "API_READONLY", "--revision", revision]
        )
    )


if __name__ == "__main__":
    main()
