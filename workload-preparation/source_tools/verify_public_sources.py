"""Inspect public publisher metadata without downloading or executing dataset code."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


def fetch_json(url):
    request = Request(
        url,
        headers={
            "User-Agent": "llm-serving-workloads-source-audit/1",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=25) as response:
        return json.load(response)


def check(source):
    result = {
        "source_id": source["source_id"],
        "source_url": source["source_url"],
        "independent_source_id": source["independent_source_id"],
        "asset_kind": source["asset_kind"],
        "materialization_status": "NOT_DOWNLOADED",
    }
    try:
        repo = source["repo_id"]
        if source["platform"] == "hf":
            url = (
                "https://huggingface.co/api/datasets/"
                + quote(repo, safe="/")
                + "?blobs=true"
            )
            data = fetch_json(url)
            card = data.get("cardData") or {}
            files = data.get("siblings") or []
            sizes = [f.get("size", (f.get("lfs") or {}).get("size")) for f in files]
            known_sizes = [s for s in sizes if isinstance(s, int)]
            info = card.get("dataset_info") or {}
            infos = info if isinstance(info, list) else [info]
            feature_names = sorted(
                {
                    f.get("name", "")
                    for i in infos
                    if isinstance(i, dict)
                    for f in i.get("features", [])
                    if isinstance(f, dict)
                }
            )
            result.update(
                {
                    "metadata_status": "PUBLISHER_METADATA_VERIFIED",
                    "metadata_url": url,
                    "resolved_repo_id": data.get("id"),
                    "revision": data.get("sha"),
                    "gated": data.get("gated"),
                    "private": data.get("private"),
                    "reported_data_license": card.get("license"),
                    "repository_file_count": len(files),
                    "known_file_size_count": len(known_sizes),
                    "known_bytes_lower_bound": sum(known_sizes),
                    "all_file_sizes_known": len(sizes) == len(known_sizes),
                    "declared_feature_names": feature_names,
                    "declared_dataset_info": info,
                    "publisher_last_modified": data.get("lastModified"),
                }
            )
        else:
            url = "https://api.github.com/repos/" + quote(repo, safe="/")
            data = fetch_json(url)
            result.update(
                {
                    "metadata_status": "PUBLISHER_METADATA_VERIFIED",
                    "metadata_url": url,
                    "resolved_repo_id": data.get("full_name"),
                    "default_branch": data.get("default_branch"),
                    "revision": None,
                    "code_license": (data.get("license") or {}).get("spdx_id"),
                    "reported_data_license": None,
                    "data_license_status": "ASSET_SPECIFIC_REVIEW_REQUIRED",
                    "archived": data.get("archived"),
                    "publisher_last_pushed": data.get("pushed_at"),
                    "repository_size_kib": data.get("size"),
                    "size_boundary": "git repository size is not dataset download size",
                }
            )
    except HTTPError as error:
        result.update({"metadata_status": "HTTP_ERROR", "http_status": error.code})
        if source["platform"] == "github" and error.code in {403, 429}:
            # Anonymous API limits do not make an otherwise public repository unavailable.
            try:
                completed = subprocess.run(
                    ["git", "ls-remote", "https://github.com/" + repo + ".git", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )
                revision = completed.stdout.split()[0]
                if not re.fullmatch(r"[0-9a-f]{40}", revision):
                    raise ValueError("invalid Git ref")
                result.update(
                    {
                        "metadata_status": "GIT_REF_VERIFIED",
                        "revision": revision,
                        "verification_method": "anonymous git ls-remote HEAD",
                        "api_http_status": error.code,
                        "data_license_status": "ASSET_SPECIFIC_REVIEW_REQUIRED",
                    }
                )
            except (
                subprocess.SubprocessError,
                OSError,
                ValueError,
                IndexError,
            ) as fallback_error:
                result["fallback_error_type"] = type(fallback_error).__name__
    except (URLError, TimeoutError, ValueError, OSError) as error:
        result.update(
            {
                "metadata_status": "METADATA_UNAVAILABLE",
                "error_type": type(error).__name__,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/catalogs/authoritative_datasets_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; use a fresh output or --overwrite")
    raw = args.catalog.read_bytes()
    sources = json.loads(raw)["sources"]
    if len({s["source_id"] for s in sources}) != len(sources):
        parser.error("source IDs must be unique")
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as pool:
        results = list(pool.map(check, sources))
    counts = dict(Counter(r["metadata_status"] for r in results))
    result = {
        "audit_version": "public-source-metadata/v1",
        "catalog_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "publisher metadata only; not actual data validation, license approval, field suitability or statistical independence certification",
        "source_count": len(sources),
        "independent_lineage_count": len({s["independent_source_id"] for s in sources}),
        "metadata_status_counts": counts,
        "sources": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"source_count": len(sources), "status_counts": counts}))


if __name__ == "__main__":
    main()
