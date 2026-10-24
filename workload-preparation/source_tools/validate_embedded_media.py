"""Check embedded image payload presence in a registered dataset; no inference/analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from vllm_hust_workload_data.registry import source_spec
from vllm_hust_workload_data.assets import asset_path, verify_file
from vllm_hust_workload_data.assets import sha256 as sha

ROOT = Path(__file__).resolve().parents[1]


def validate_file(path):
    file = pq.ParquetFile(path)
    if "image" not in file.schema_arrow.names:
        raise ValueError("expected image column")
    rows, nonempty, byte_count = 0, 0, 0
    for batch in file.iter_batches(batch_size=256, columns=["image"]):
        images = batch.column(0)
        data = images.field("bytes")
        lengths = pc.binary_length(data)
        rows += batch.num_rows
        nonempty += (
            pc.sum(
                pc.cast(pc.fill_null(pc.greater(lengths, 0), False), "int64")
            ).as_py()
            or 0
        )
        byte_count += pc.sum(lengths).as_py() or 0
    return {
        "rows": rows,
        "nonempty_embedded_images": nonempty,
        "embedded_image_bytes": byte_count,
        "all_rows_have_embedded_image": rows == nonempty,
        "file_sha256": sha(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a fresh validation receipt")
    source = source_spec(args.source_id)
    selected = [f for f in source["files"] if f["path"].endswith(".parquet")]
    if not selected:
        parser.error("source has no pinned Parquet files")
    files = []
    for file in selected:
        verify_file(args.data_root, file)
        files.append(
            {
                "path": file["path"],
                **validate_file(asset_path(args.data_root, file["path"])),
            }
        )
    report = {
        "version": "embedded-media-validation/v1",
        "source_id": args.source_id,
        "revision": source["revision"],
        "scope": "Embedded bytes present, not independent-image deduplication, image semantic quality, visual decoding or model evaluation.",
        "files": files,
        "rows": sum(f["rows"] for f in files),
        "all_rows_have_embedded_image": bool(files)
        and all(f["all_rows_have_embedded_image"] for f in files),
        "script_sha256": sha(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_id": args.source_id,
                "rows": report["rows"],
                "all_rows_have_embedded_image": report["all_rows_have_embedded_image"],
            }
        )
    )
    if not report["all_rows_have_embedded_image"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
