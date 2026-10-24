"""Explicit read-only adapter validation. Aggregate-only output; no raw payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_hust_workload_data import iter_workload, source_spec
from vllm_hust_workload_data.assets import asset_path, sha256


def validate(source_id, data_root, *, limit_per_file=None):
    manifest = source_spec(source_id)
    files = []
    for file in manifest["files"]:
        if file["role"] != "data":
            continue
        count = 0
        unknown_fields = set()
        kinds = set()
        for record in iter_workload(
            source_id, data_root=data_root, files=[file["path"]], limit=limit_per_file
        ):
            count += 1
            unknown_fields.update(record.replay["unmapped_field_names"])
            kinds.add(record.runtime["kind"])
            # Validate envelope separation without serializing sensitive content.
            assert "outcome" not in record.input_record()
            if record.runtime["kind"] == "request_replay":
                assert record.request_body()["messages"]
        after_hash = sha256(asset_path(data_root, file["path"]))
        if after_hash != file["sha256"]:
            raise ValueError("source changed during read-only validation")
        files.append(
            {
                "path": file["path"],
                "sha256": after_hash,
                "rows_adapted": count,
                "source_rows_recorded": file.get("rows"),
                "kinds": sorted(kinds),
                "unmapped_field_count": len(unknown_fields),
                "unmapped_field_names": sorted(unknown_fields)
                if not source_id.startswith("PAIO-")
                else "SUPPRESSED_FOR_PRIVATE_SOURCE",
                "scope": "FULL_FILE"
                if limit_per_file is None
                else "BOUNDED_PREFIX_NOT_FULL_FILE",
            }
        )
    return {
        "source_id": source_id,
        "revision": manifest["revision"],
        "files": files,
        "status": "ADAPTER_READ_VALIDATED",
        "validation_level": "source_fields_and_mapping_not_runtime_reproduction",
        "model_calls": 0,
        "synthetic_outcomes": 0,
        "rows_adapted": sum(f["rows_adapted"] for f in files),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit-per-file", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error("output exists; choose a fresh receipt path")
    result = validate(args.source, args.data_root, limit_per_file=args.limit_per_file)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "source_id": args.source,
                "rows_adapted": result["rows_adapted"],
                "status": result["status"],
            }
        )
    )
