"""Fetch explicitly selected, pinned public task files into ignored local storage.

Never accepts gates, downloads executable code, or evaluates generated programs.
Parquet validation uses an already available optional pyarrow installation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen

from verify_public_sources import fetch_json


ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = {
    "DOLLY": ["databricks-dolly-15k.jsonl"],
    "GSM8K": ["main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"],
    "MMLU_PRO": [
        "data/test-00000-of-00001.parquet",
        "data/validation-00000-of-00001.parquet",
    ],
    "AI2_ARC": [
        f"{part}/{split}-00000-of-00001.parquet"
        for part in ["ARC-Challenge", "ARC-Easy"]
        for split in ["train", "validation", "test"]
    ],
    "BIGCODEBENCH": ["data/v0.1.4-00000-of-00001.parquet"],
    "SCIENCEAGENTBENCH": ["data/verified-00000-of-00001.parquet"],
    "DS1000": ["test.jsonl"],
    "STRUCTEVAL": ["data/train-00000-of-00001.parquet"],
}


def destination(base: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    if parts.is_absolute() or any(p in {"..", "."} or ":" in p for p in parts.parts):
        raise ValueError("unsafe publisher path")
    result = (base / Path(*parts.parts)).resolve()
    if not result.is_relative_to(base.resolve()):
        raise ValueError("publisher path escapes local root")
    return result


def inspect_data(path: Path) -> dict:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        file = pq.ParquetFile(path)
        count = sum(batch.num_rows for batch in file.iter_batches(batch_size=1024))
        return {
            "row_count": count,
            "columns": file.schema_arrow.names,
            "all_batches_decoded": count == file.metadata.num_rows,
        }
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("expected a JSON array of object records")
        return {
            "row_count": len(rows),
            "columns": sorted({key for row in rows for key in row}),
            "all_batches_decoded": True,
        }
    count = 0
    fields = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            def reject_constant(value):
                raise ValueError("non-finite JSON literal")

            row = json.loads(line, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError("expected object records")
            fields.update(row)
            count += 1
    return {"row_count": count, "columns": sorted(fields), "all_batches_decoded": True}


def acquire(
    source: dict, metadata: dict, selections=None, max_source_bytes=20_000_000
) -> dict:
    repo = source["repo_id"]
    revision = metadata.get("revision")
    result = {
        "source_id": source["source_id"],
        "independent_source_id": source["independent_source_id"],
        "repo_id": repo,
        "revision": revision,
        "source_url": source["source_url"],
        "scope": "selected named task splits, not all versions, media or execution environments",
    }
    if metadata.get("gated") is not False or not revision:
        return {**result, "status": "NOT_PUBLIC_OR_UNPINNED"}
    data = fetch_json(
        "https://huggingface.co/api/datasets/" + quote(repo, safe="/") + "?blobs=true"
    )
    if data.get("sha") != revision:
        return {**result, "status": "PUBLISHER_REVISION_CHANGED"}
    files = {f["rfilename"]: f for f in data["siblings"]}
    selected = (SELECTIONS if selections is None else selections)[source["source_id"]]
    if not selected or not all(
        name.endswith((".json", ".jsonl", ".parquet")) for name in selected
    ):
        raise ValueError("only explicitly selected data files are allowed")
    if any(name not in files for name in selected):
        return {**result, "status": "SELECTED_FILE_MISSING"}
    if sum(files[name].get("size", 10**12) for name in selected) > max_source_bytes:
        return {**result, "status": "SIZE_REVIEW_REQUIRED"}
    base = ROOT / ".cache/public_datasets" / source["source_id"] / revision
    base.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name in ["README.md", *selected]:
        if name not in files:
            continue
        info = files[name]
        path = destination(base, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://huggingface.co/datasets/"
            + quote(repo, safe="/")
            + "/resolve/"
            + revision
            + "/"
            + quote(name, safe="/")
        )
        expected_size = info.get("size")
        if not isinstance(expected_size, int) or expected_size > max_source_bytes:
            raise ValueError("file size unavailable or beyond bounded acquisition")
        if not path.exists():
            temporary = path.with_name(path.name + ".part")
            received = 0
            with (
                urlopen(
                    Request(
                        url,
                        headers={"User-Agent": "llm-serving-workloads-source-audit/1"},
                    ),
                    timeout=45,
                ) as response,
                temporary.open("wb") as handle,
            ):
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > expected_size:
                        raise ValueError("download exceeds publisher size")
                    handle.write(block)
            if received != expected_size:
                raise ValueError("download size differs from publisher metadata")
            temporary.replace(path)
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        expected_sha = (info.get("lfs") or {}).get("sha256")
        if len(raw) != expected_size or (expected_sha and sha != expected_sha):
            raise ValueError("local asset checksum mismatch")
        git_blob = info.get("blobId")
        if not expected_sha and git_blob:
            actual_blob = hashlib.sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw
            ).hexdigest()
            if actual_blob != git_blob:
                raise ValueError("Git blob identity mismatch")
        entry = {
            "file": name,
            "relative_local_path": path.relative_to(ROOT).as_posix(),
            "download_url": url,
            "bytes": len(raw),
            "sha256": sha,
            "publisher_lfs_sha256": expected_sha,
            "publisher_git_blob": git_blob,
        }
        if name in selected:
            entry.update(inspect_data(path))
        outputs.append(entry)
    return {
        **result,
        "status": "SELECTED_ASSETS_DOWNLOADED_AND_PARSED",
        "reported_data_license": metadata.get("reported_data_license"),
        "files": outputs,
        "row_count": sum(f.get("row_count", 0) for f in outputs),
        "excluded_variants": [
            name
            for name in files
            if name.endswith((".jsonl", ".parquet", ".csv")) and name not in selected
        ],
        "quality_or_runtime_evaluation": "NOT_RUN",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--selections",
        type=Path,
        help="JSON source_id -> explicit publisher filenames; no code",
    )
    parser.add_argument("--max-source-bytes", type=int, default=20_000_000)
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.error("output exists; use --overwrite or a new path")
    catalog = json.loads(
        (ROOT / "data/catalogs/authoritative_datasets_v1.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = {
        s["source_id"]: s
        for s in json.loads(args.metadata.read_text(encoding="utf-8"))["sources"]
    }
    selections = (
        json.loads(args.selections.read_text(encoding="utf-8"))
        if args.selections
        else SELECTIONS
    )
    if not isinstance(selections, dict) or not set(selections) <= {
        s["source_id"] for s in catalog["sources"]
    }:
        parser.error("invalid selections or unknown sources")
    selected = [s for s in catalog["sources"] if s["source_id"] in selections]

    def attempt(source):
        try:
            return acquire(
                source, metadata[source["source_id"]], selections, args.max_source_bytes
            )
        except Exception as error:
            return {
                "source_id": source["source_id"],
                "status": "ACQUISITION_FAILED",
                "error_type": type(error).__name__,
            }

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, selected))
    report = {
        "version": "public-asset-acquisition/v1",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
        "sources": results,
        "selected_source_count": len(selected),
        "successful_source_count": sum(
            s["status"] == "SELECTED_ASSETS_DOWNLOADED_AND_PARSED" for s in results
        ),
        "row_count": sum(s.get("row_count", 0) for s in results),
        "scope": "raw selected files remain git-ignored; row counts are not claims of unique independent tasks or completed task/runtime analysis",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "selected_source_count",
                    "successful_source_count",
                    "row_count",
                ]
            }
        )
    )
    if report["successful_source_count"] != report["selected_source_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
