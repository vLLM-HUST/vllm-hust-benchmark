"""Bind audited original files and prepare source references, never model tasks.

Reuse reviewed input/outcome selectors; research feature views are deliberately
empty. Original data stays external. Native execution/scoring is a separate gate.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPINGS = {
    "DUREADER": ROOT / "configs/workload-native/reused-source-mappings/dureader.json",
    "LONGALPACA": ROOT
    / "configs/workload-native/reused-source-mappings/longalpaca.json",
    "NATURALQUESTIONS": ROOT
    / "configs/workload-native/reused-source-mappings/naturalquestions.json",
}


def prepare(source_id, *, data_root, output, files=None, limit=None, catalog_root=None):
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.mapping_contract import load_mapping
    from vllm_hust_workload_data.pipeline import run
    from vllm_hust_workload_data.registry import source_spec

    if source_id not in MAPPINGS:
        raise ValueError("source preparation not implemented; no generic fallback")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("explicit positive preparation limit required")
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError("explicit original data root required")
    if (
        output.exists()
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from original data root required")
    catalog_root = (
        ROOT / "workload-preparation" if catalog_root is None else Path(catalog_root)
    )
    mapping = load_mapping(MAPPINGS[source_id])
    if mapping["source"]["source_id"] != source_id or mapping["features"]:
        raise ValueError("matching data-only source mapping required")
    source = source_spec(source_id, catalog_root)
    if source["revision"] != mapping["source"]["revision"]:
        raise ValueError(
            "catalog revision differs from reviewed mapping; re-review required"
        )
    registered = {
        item["path"] for item in source["files"] if item["role"] == "raw_source"
    }
    selected = (
        [item["path"] for item in source["files"] if item["role"] == "raw_source"]
        if files is None
        else files
    )
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(name not in registered for name in selected)
    ):
        raise ValueError("select original registered raw-source files only")
    bound = bind_source(
        source_id,
        data_root=data_root,
        files=selected,
        revision=mapping["source"]["revision"],
        mapping=mapping,
        catalog_root=catalog_root,
    )
    output.mkdir(mode=0o700, parents=True)
    receipt = {
        "source_id": source_id,
        "status": "PREPARING_SOURCE_REFERENCES",
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "native_execution_complete": False,
        "raw_data_copied": False,
        "research_observation_or_feature_views_migrated": False,
        "selected_original_files": selected,
        "source_rows_fully_bound": sum(item["rows"] for item in bound["files"]),
        "selected_output_limit": limit,
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )

    save()
    try:
        manifest = output / "source-manifest.private.json"
        profile = output / "source-mapping.private.json"
        manifest.write_text(json.dumps(bound, indent=2) + "\n", encoding="utf-8")
        profile.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
        entry = {
            "source_id": source_id,
            "data_root": str(data_root),
            "files": selected,
            "manifest": str(manifest),
            "mapping": str(profile),
        }
        if limit is not None:
            entry["limit"] = limit
        config = {"sources": [entry]}
        (output / "config.private.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        result = run(config, output=output / "prepared", catalog_root=catalog_root)
        receipt.update(
            status="ORIGINAL_RECORD_REFERENCES_PREPARED_NOT_NATIVE_DELIVERY",
            preparation=result,
        )
    except Exception as exc:
        receipt.update(
            status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_id", choices=sorted(MAPPINGS))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", nargs="+")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.source_id,
                data_root=args.data_root,
                output=args.output,
                files=args.files,
                limit=args.limit,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
