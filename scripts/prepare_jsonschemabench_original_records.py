"""Prepare exact official schema-row references with existing public mapping APIs.

Opt-in official snapshot view; leaves legacy three-schema catalog/readers intact.
No model calls, invented prompts, schema normalization or research observations.
"""

import argparse
import hashlib
import json
from pathlib import Path

from prepare_jsonschemabench_original_assets import PINS
from prepare_xsum_original_assets import bound_path, verify

MAPPING = PINS.with_name("jsonschemabench-official-source-mapping.json")


def prepare(*, data_root, output, files=None):
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.mapping_contract import load_mapping
    from vllm_hust_workload_data.pipeline import run

    data_root = Path(data_root).resolve(strict=True)
    output = Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError("fresh output separate from original snapshot required")
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    mapping = load_mapping(MAPPING)
    if (
        mapping["source"]["revision"] != pins["dataset"]["revision"]
        or mapping["features"]
    ):
        raise ValueError("fixed matching data-only mapping required")
    available = {
        item["path"]: item
        for item in pins["dataset"]["files"]
        if item["path"].endswith(".parquet")
    }
    selected = list(available) if files is None else list(files)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= set(available)
    ):
        raise ValueError("explicit distinct original registered Parquet files required")
    for name in selected:
        verify(bound_path(data_root, "", available[name]), available[name])
    bound = bind_source(
        "JSONSCHEMABENCH",
        data_root=data_root,
        files=selected,
        revision=mapping["source"]["revision"],
        mapping=mapping,
    )
    # The generic binder alone observes local hashes. This source-specific entry
    # additionally authenticates every original file against the publisher pins.
    for item in bound["files"]:
        original = available[item["path"]]
        if item["sha256"] != original["sha256"] or item["bytes"] != original["bytes"]:
            raise ValueError("original file identity changed during binding")
        item["identity_assurance"] = "matches_pinned_publisher_file_identity"
    bound.update(
        upstream_revision_verified=True,
        publisher_asset_pins_sha256=hashlib.sha256(PINS.read_bytes()).hexdigest(),
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    manifest = output / "source-manifest.private.json"
    profile = output / "source-mapping.private.json"
    manifest.write_text(json.dumps(bound, indent=2) + "\n", encoding="utf-8")
    profile.write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    config = {
        "sources": [
            {
                "source_id": "JSONSCHEMABENCH",
                "data_root": str(data_root),
                "files": selected,
                "manifest": str(manifest),
                "mapping": str(profile),
            }
        ]
    }
    (output / "config.private.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    result = run(config, output=output / "prepared")
    receipt = {
        "source_id": "JSONSCHEMABENCH",
        "status": "ORIGINAL_OFFICIAL_SCHEMA_ROW_REFERENCES_PREPARED_NOT_NATIVE_DELIVERY",
        "revision": mapping["source"]["revision"],
        "original_files": len(selected),
        "original_rows_in_selected_views": sum(item["rows"] for item in bound["files"]),
        "raw_json_schema_and_original_unique_id_preserved": True,
        "legacy_catalog_or_three_schema_assets_modified": False,
        "view_rows_are_not_unique_independent_tasks": True,
        "raw_data_copied": False,
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "preparation": result,
    }
    (output / "receipt-public.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(data_root=args.data_root, output=args.output, files=args.files)
        )
    )
