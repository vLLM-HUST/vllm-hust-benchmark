"""Website and Release navigation; explicit remote refresh, offline reproducibility."""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen
from urllib.error import URLError

from data_index import ROOT, render, tracked_paths

SOURCES = "docs/data-entrypoints-sources.json"
OUTPUT = "docs/data-entrypoints.json"
REPO = "vLLM-HUST/vllm-hust-benchmark"
WEBSITE = "vLLM-HUST/vllm-hust-website"
PREFIX = "website-publication/data/"
URL = re.compile(r"https?://[^\s\"'<>`\\]+")
LOCAL = re.compile(r"[\"'`]((?:\./|/)?data/[^\s\"'<>`]+)")
BENCHMARK_PATH = re.compile(
    r"[\"'`]((?:reports|archive|submissions|leaderboard-data|website-publication)/[^\s\"'<>`]+)"
)
MANIFEST_URL = "https://vllm-hust.sage.org.ai/data/benchmark-source-manifest.json"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def references(text):
    # Literal inventory only: no execution or pretending template expressions resolve.
    def strings(value):
        if isinstance(value, str):
            yield '"' + value + '"'
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)

    # Decode JSON escapes (including Unicode and escaped slashes) before scanning.
    # Non-JSON frontend/Markdown stays a literal text scan, never executable input.
    try:
        chunks = strings(json.loads(text))
    except ValueError:
        chunks = [text]
    found = set()
    for chunk in chunks:
        found.update(URL.findall(chunk))
        found.update(LOCAL.findall(chunk))
        found.update(BENCHMARK_PATH.findall(chunk))
    return sorted(found)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def refresh(website, revision):
    commit = (
        git(website, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    )
    paths = git(website, "ls-tree", "-rz", "--name-only", commit).decode().split("\0")
    consumers = {}
    inputs = {}
    for path in sorted(paths):
        # Public frontend only, not tests, build tools, or historic screenshots.
        if not (
            path.endswith((".html", ".js", ".mjs"))
            and ("/" not in path or path.startswith("assets/"))
        ):
            continue
        raw = git(website, "show", f"{commit}:{path}")
        inputs[path] = sha(raw)
        for target in references(raw.decode("utf-8")):
            consumers.setdefault(target, []).append(path)
    # Production assembles data at deploy time; the website Git copy is historical.
    manifest_source = MANIFEST_URL
    manifest_error = None
    try:
        with urlopen(
            Request(
                MANIFEST_URL,
                headers={"User-Agent": "curl/8.0 benchmark-navigation-audit"},
            ),
            timeout=20,
        ) as response:
            manifest = json.load(response)
    except (URLError, TimeoutError) as error:
        manifest_error = str(error)
        manifest_source = f"https://github.com/{WEBSITE}/blob/{commit}/data/benchmark-source-manifest.json"
        manifest = json.loads(
            git(website, "show", f"{commit}:data/benchmark-source-manifest.json")
        )
    if (
        manifest.get("schema") != "benchmark-website-publication/v1"
        or manifest.get("repository") != REPO
    ):
        raise ValueError("Unexpected production publication manifest")
    pages = json.loads(
        subprocess.check_output(
            [
                "gh",
                "api",
                "--paginate",
                "--slurp",
                f"repos/{REPO}/releases?per_page=100",
            ]
        )
    )
    releases = []
    for page in pages:
        for release in page:
            if release["draft"]:
                continue
            # The releases list can omit incomplete uploads from embedded assets.
            # Enumerate each attachment endpoint; retain failed uploads as metadata,
            # never silently advertise them as downloadable evidence.
            asset_pages = json.loads(
                subprocess.check_output(
                    [
                        "gh",
                        "api",
                        "--paginate",
                        "--slurp",
                        f"repos/{REPO}/releases/{release['id']}/assets?per_page=100",
                    ]
                )
            )
            assets = [asset for asset_page in asset_pages for asset in asset_page]
            releases.append(
                {
                    "tag": release["tag_name"],
                    "url": release["html_url"],
                    "id": release["id"],
                    "published_at": release["published_at"],
                    "assets": sorted(
                        [
                            {
                                "name": a["name"],
                                "url": a["browser_download_url"],
                                "id": a["id"],
                                "bytes": a["size"],
                                "digest": a.get("digest"),
                                "updated_at": a["updated_at"],
                                "state": a["state"],
                            }
                            for a in assets
                        ],
                        key=lambda a: a["name"],
                    ),
                }
            )
    return {
        "schema": "benchmark-navigation-sources/v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "release_repository": REPO,
        "release_scope": "All non-draft Releases and separately paginated attachment metadata, including incomplete uploads. Only state=uploaded denotes a completed attachment; no asset download or checksum verification.",
        "releases": sorted(releases, key=lambda r: r["tag"]),
        "website_repository": WEBSITE,
        "website_commit": commit,
        "website_scope": "Literal HTTP(S), data/ and benchmark evidence paths in root HTML/JS/MJS and assets HTML/JS/MJS; dynamic expressions are not evaluated. Source snapshot is independent of production deployment; not a browser audit.",
        "website_inputs_sha256": inputs,
        "website_references": [
            {"target": t, "consumers": consumers[t]} for t in sorted(consumers)
        ],
        "website_publication_manifest": manifest,
        "website_publication_manifest_url": manifest_source,
        "production_manifest_error": manifest_error,
    }


def publication_bytes(root, path, tracked):
    raw = (root / path).read_bytes()
    if not path.endswith(".publication.json"):
        return raw, [path]
    descriptor = json.loads(raw)
    if descriptor["encoding"] != "gzip":
        raise ValueError(f"Unsupported publication encoding: {path}")
    chunks, inputs = [], [path]
    for part in descriptor["parts"]:
        name = part["name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Unsafe publication part")
        source = str(Path(path).with_name(name))
        if source not in tracked:
            raise ValueError(f"Untracked publication part: {source}")
        chunk = (root / source).read_bytes()
        if sha(chunk) != part["sha256"] or len(chunk) != part["bytes"]:
            raise ValueError(f"Publication part checksum mismatch: {source}")
        chunks.append(chunk)
        inputs.append(source)
    return gzip.decompress(b"".join(chunks)), inputs


def build(root, paths, sources):
    tracked = set(paths)
    bindings, refs, inputs = [], {}, {}
    published = {
        f["website_path"]: f for f in sources["website_publication_manifest"]["files"]
    }
    for path in sorted(tracked):
        if not path.startswith(PREFIX) or ".publication.part-" in path:
            continue
        raw, used = publication_bytes(root, path, tracked)
        inputs.update({p: sha((root / p).read_bytes()) for p in used})
        website_path = "data/" + path.removeprefix(PREFIX).removesuffix(
            ".publication.json"
        )
        old = published.get(website_path)
        bindings.append(
            {
                "website_path": website_path,
                "benchmark_path": path,
                "storage_paths": used,
                "sha256": sha(raw),
                "snapshot_publication_status": "same-bytes"
                if old and old["sha256"] == sha(raw)
                else "changed-since-snapshot"
                if old
                else "not-in-snapshot",
            }
        )
        if website_path.endswith((".json", ".jsonl", ".md", ".txt", ".csv")):
            for target in references(raw.decode("utf-8")):
                # Keep evidence links rather than every external dataset/model homepage.
                if not urlsplit(target).scheme or "vllm-hust" in target.lower():
                    refs.setdefault(target, set()).add(path)
    for ref in sources["website_references"]:
        refs.setdefault(ref["target"], set()).update(
            "website:" + p for p in ref["consumers"]
        )
    by_website = {b["website_path"]: b["benchmark_path"] for b in bindings}
    links = []
    for target in sorted(refs):
        parsed = urlsplit(target)
        local = parsed.path.removeprefix("./").removeprefix("/")
        record = {"target": target, "consumers": sorted(refs[target])}
        if local.startswith("data/") and (
            not parsed.scheme or parsed.netloc == "vllm-hust.sage.org.ai"
        ):
            record["benchmark_path"] = by_website.get(unquote(local))
            record["resolution"] = (
                "mapped"
                if record["benchmark_path"]
                else "generated-publication-manifest"
                if local == "data/benchmark-source-manifest.json"
                else "directory-or-dynamic"
                if local.endswith("/") or "${" in local
                else "unmapped-frontend-reference"
                if any(c.startswith("website:") for c in refs[target])
                else "unmapped-embedded-path"
            )
        elif not parsed.scheme:
            record["benchmark_path"] = unquote(local)
            record["resolution"] = (
                "tracked-evidence-path"
                if unquote(local) in tracked
                else "tracked-evidence-directory"
                if any(p.startswith(unquote(local).rstrip("/") + "/") for p in tracked)
                else "not-in-current-checkout"
            )
        else:
            record["resolution"] = (
                "dynamic-template-not-evaluated"
                if "${" in target
                else "literal-external-link-not-fetched"
            )
        links.append(record)
    releases = [
        {
            **release,
            "verification": "API metadata only; verify downloaded bytes against custody manifests before use",
        }
        for release in sources["releases"]
    ]
    return {
        "schema": "benchmark-data-entrypoints/v1",
        "source_snapshot": SOURCES,
        "captured_at": sources["captured_at"],
        "website_commit": sources["website_commit"],
        "published_benchmark_commit": sources["website_publication_manifest"]["commit"],
        "publication_manifest_source": sources["website_publication_manifest_url"],
        "production_manifest_error": sources["production_manifest_error"],
        "limits": "Offline navigation, not browser validation or evidence acceptance. Remote inventory is a dated snapshot. Literal links may be historical, dynamic, or unavailable. Reconstructed publication parts are checked locally; Release asset bytes are not downloaded.",
        "publication_inputs_sha256": inputs,
        "publication_bindings": bindings,
        "snapshot_only_website_paths": sorted(set(published) - set(by_website)),
        "references": links,
        "releases": releases,
        "counts": {
            "publication_bindings": len(bindings),
            "references": len(links),
            "releases": len(releases),
            "assets": sum(len(r["assets"]) for r in releases),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--refresh",
        action="store_true",
        help="Read pinned website Git objects and GitHub Release metadata",
    )
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--website", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--query", default="")
    parser.add_argument(
        "--kind",
        choices=("references", "publication_bindings", "releases", "assets"),
        default="references",
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    if args.limit < 1 or (args.refresh and (not args.website or not args.revision)):
        parser.error(
            "Positive limit required; --refresh requires --website and --revision"
        )
    if args.refresh:
        sources = refresh(args.website, args.revision)
        previous_path = ROOT / SOURCES
        if previous_path.exists():
            previous = json.loads(previous_path.read_text())
            if previous.get("releases") and not sources["releases"]:
                raise ValueError(
                    "Remote refresh returned no public Releases; refusing to erase the existing inventory. Verify the API response and retry."
                )
        (ROOT / SOURCES).write_text(render(sources))
    else:
        sources = json.loads((ROOT / SOURCES).read_text())
    catalog = build(ROOT, tracked_paths(ROOT), sources)
    if args.refresh or args.write:
        (ROOT / OUTPUT).write_text(render(catalog))
        print(render(catalog["counts"]))
        return 0
    if args.check:
        if not (ROOT / OUTPUT).exists() or (ROOT / OUTPUT).read_text() != render(
            catalog
        ):
            print(
                "Entry-point index is stale: python scripts/data_entrypoints.py --write"
            )
            return 1
        print(
            "Website/Release navigation is current against the recorded snapshot; no live availability claim"
        )
        return 0
    terms = args.query.casefold().split()
    candidates = (
        [
            {"release_tag": r["tag"], **asset}
            for r in catalog["releases"]
            for asset in r["assets"]
        ]
        if args.kind == "assets"
        else catalog[args.kind]
    )
    result = [
        r
        for r in candidates
        if all(t in json.dumps(r, ensure_ascii=False).casefold() for t in terms)
    ]
    print(
        render(
            {
                "matched": len(result),
                "truncated": len(result) > args.limit,
                "results": result[: args.limit],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
