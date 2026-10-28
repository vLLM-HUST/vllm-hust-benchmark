"""Prepare an explicit original FLORES direction; no translation or scoring."""

import argparse
import json
from pathlib import Path


def prepare(*, data_root, output, source_language, target_language, catalog_root=None):
    from vllm_hust_workload_data.flores200 import translation_mapping
    from vllm_hust_workload_data.bindings import bind_source
    from vllm_hust_workload_data.pipeline import run
    from vllm_hust_workload_data.registry import source_spec

    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if (
        output.exists()
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError("fresh private output separate from original archive required")
    mapping = translation_mapping(
        source_language, target_language, catalog_root=catalog_root
    )
    source = source_spec("FLORES200", catalog_root)
    manifest = bind_source(
        "FLORES200",
        data_root=data_root,
        files=["flores200_dataset.tar.gz"],
        revision=source["revision"],
        mapping=mapping,
        catalog_root=catalog_root,
    )
    output.mkdir(mode=0o700, parents=True)
    mapping_path, manifest_path = (
        output / "mapping.private.json",
        output / "source-manifest.private.json",
    )
    mapping_path.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config = {
        "sources": [
            {
                "source_id": "FLORES200",
                "data_root": str(data_root),
                "mapping": str(mapping_path),
                "manifest": str(manifest_path),
            }
        ]
    }
    (output / "config.private.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    receipt = {
        "source_id": "FLORES200",
        "status": "PREPARING_ORIGINAL_DIRECTION_REFERENCES",
        "source_language": source_language,
        "target_language": target_language,
        "original_combined_split_rows": manifest["files"][0]["rows"],
        "new_model_calls": 0,
        "source_checkbox_eligible": False,
        "raw_payload_published": False,
        "research_analysis_migrated": False,
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        result = run(config, output=output / "prepared", catalog_root=catalog_root)
        receipt.update(
            status="ORIGINAL_DIRECTION_REFERENCES_PREPARED_NOT_NATIVE_DELIVERY",
            preparation=result,
        )
    except Exception as error:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(error).__name__)
        raise
    finally:
        save()
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                data_root=args.data_root,
                output=args.output,
                source_language=args.source_language,
                target_language=args.target_language,
            )
        )
    )


if __name__ == "__main__":
    main()
