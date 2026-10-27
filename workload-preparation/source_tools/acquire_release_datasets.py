"""Acquire explicit hash-pinned publisher releases; validate records, never run them."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "data/catalogs/release_dataset_selection_v1.json"


def failure_metadata(error):
    """Keep actionable failure types without exporting URLs, paths or credentials."""
    result = {"error_type": type(error).__name__}
    if isinstance(error, HTTPError):
        result["http_status"] = error.code
    elif isinstance(error, URLError):
        result["reason_type"] = type(error.reason).__name__
    return result


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path, max_uncompressed_bytes=8_000_000_000):
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            columns = next(reader)
            if not columns or len(set(columns)) != len(columns):
                raise ValueError("invalid CSV header")
            count = 0
            for row in reader:
                if not row:
                    continue
                if len(row) != len(columns):
                    raise ValueError("CSV row width mismatch")
                count += 1
    elif path.name.endswith(".jsonl.gz"):
        columns, count, decoded = set(), 0, 0
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                decoded += len(line.encode("utf-8"))
                if decoded > max_uncompressed_bytes:
                    raise ValueError("uncompressed data exceeds declared safety bound")
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("non-object trace record")
                columns.update(row)
                count += 1
        columns = sorted(columns)
    else:
        raise ValueError("unsupported release format")
    return {"row_count": count, "columns": columns, "all_batches_decoded": True}


def acquire(source, *, data_root=None):
    # Selection is a local reviewed allowlist, not filenames extracted from an archive.
    if any(
        Path(x).name != x or x in {".", ".."} or ":" in x
        for x in [
            source["source_id"],
            source["tag"],
            *[f["name"] for f in source["files"]],
        ]
    ):
        raise ValueError("unsafe release path")
    explicit_root = data_root is not None
    cache_root = (
        Path(data_root).resolve() if explicit_root else ROOT / ".cache/public_datasets"
    )
    if explicit_root and cache_root.is_relative_to(ROOT.resolve()):
        raise ValueError("explicit data root must be outside the component checkout")
    base = cache_root / source["source_id"] / source["tag"]
    base.mkdir(parents=True, exist_ok=True)
    files = []
    for entry in source["files"]:
        path = base / entry["name"]
        url = f"https://github.com/{source['repo_id']}/releases/download/{source['tag']}/{entry['name']}"
        if not path.exists():
            temporary = path.with_name(path.name + ".part")
            received = 0
            with (
                urlopen(
                    Request(
                        url,
                        headers={"User-Agent": "llm-serving-workloads-data-curation"},
                    ),
                    timeout=60,
                ) as response,
                temporary.open("wb") as stream,
            ):
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    received += len(block)
                    if received > entry["bytes"]:
                        raise ValueError("publisher size exceeded")
                    stream.write(block)
            if received != entry["bytes"] or sha(temporary) != entry["sha256"]:
                raise ValueError("download identity mismatch")
            temporary.replace(path)
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("existing data mismatch; refusing to overwrite")
        files.append(
            {
                "file": entry["name"],
                "relative_local_path": path.relative_to(
                    cache_root if explicit_root else ROOT
                ).as_posix(),
                "path_relative_to": "explicit_data_root"
                if explicit_root
                else "component_root_legacy_cache",
                "download_url": url,
                "bytes": entry["bytes"],
                "sha256": entry["sha256"],
                "publisher_sha256": entry["sha256"],
                **inspect(path),
            }
        )
        print(
            json.dumps(
                {
                    "source_id": source["source_id"],
                    "file": entry["name"],
                    "rows": files[-1]["row_count"],
                    "status": "HASH_AND_PARSE_VERIFIED",
                }
            ),
            flush=True,
        )
    return {
        "source_id": source["source_id"],
        "independent_source_id": source["source_id"],
        "repo_id": source["repo_id"],
        "source_url": f"https://github.com/{source['repo_id']}",
        "revision": source["tag"],
        "asset_kind": source["asset_kind"],
        "unit": source["unit"],
        "scope": source["scope"],
        "files": files,
        "row_count": sum(f["row_count"] for f in files),
        "status": "SELECTED_ASSETS_DOWNLOADED_AND_PARSED",
        "quality_or_runtime_evaluation": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["BURSTGPT", "TRACELAB"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="external server data directory; omitted only for legacy cache compatibility",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("report exists; choose a new report or explicitly overwrite")
    sources = json.loads(SELECTION.read_text(encoding="utf-8"))["sources"]
    if args.source:
        sources = [s for s in sources if s["source_id"] == args.source]

    def attempt(source):
        try:
            return acquire(source, data_root=args.data_root)
        except Exception as error:
            return {
                "source_id": source["source_id"],
                "status": "ACQUISITION_FAILED",
                **failure_metadata(error),
            }

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, sources))
    report = {
        "version": "public-release-acquisition/v1",
        "selection_sha256": sha(SELECTION),
        "script_sha256": sha(Path(__file__)),
        "sources": results,
        "successful_source_count": sum(
            s["status"] == "SELECTED_ASSETS_DOWNLOADED_AND_PARSED" for s in results
        ),
        "row_count": sum(s.get("row_count", 0) for s in results),
        "scope": "Per-source units must be retained; aggregate rows are not independent tasks. No model or downloaded code executed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "sources": len(results),
                "successful": report["successful_source_count"],
                "rows": report["row_count"],
            }
        )
    )
    if report["successful_source_count"] != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
