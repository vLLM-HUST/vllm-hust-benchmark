"""Verify publisher CPU package relations without Agent or scientific execution."""

import argparse
import json
from pathlib import Path

from vllm_hust_workload_data.naturebench import original_cpu_packages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--prepare-output",
        help="Optional fresh external reference-only index directory",
    )
    args = parser.parse_args()
    packages = []
    for row in original_cpu_packages(args.data_root):
        packages.append(
            {
                "task_id": row["task_id"],
                "problem_assets": len(row["input"]["assets"]),
                "evaluation_assets": len(row["outcome"]["evaluation_assets"]),
                "environment_assets": len(row["environment_assets"]),
                "license_assets": len(row["license_assets"]),
                "original_metadata_fields": len(
                    row["outcome"]["original_task_metadata"]
                ),
            }
        )
    indexed = None
    if args.prepare_output:
        from vllm_hust_workload_data.adapters import iter_workload
        from vllm_hust_workload_data.consumer import resolve_input
        from vllm_hust_workload_data.pipeline import prepared_rows, run

        originals = list(iter_workload("NATUREBENCH", data_root=Path(args.data_root)))
        run(
            {"sources": [{"source_id": "NATUREBENCH", "data_root": args.data_root}]},
            output=args.prepare_output,
        )
        database = Path(args.prepare_output) / "workload.sqlite"
        references = list(prepared_rows(database))
        if len(references) != len(originals):
            raise ValueError("original task count changed during preparation")
        for reference, original in zip(references, originals):
            restored = resolve_input(
                database, reference["record_id"], data_root=Path(args.data_root)
            )
            if (
                restored.input_record() != original.input_record()
                or restored.outcome_record() != original.outcome_record()
            ):
                raise ValueError(
                    "original task package changed during index resolution"
                )
        indexed = len(references)
    print(
        json.dumps(
            {
                "status": "ORIGINAL_CPU_PACKAGE_RELATIONS_VERIFIED_NOT_NATIVE_EXECUTION",
                "packages": packages,
                "unified_index_and_provider_readbacks": indexed,
                "model_calls": 0,
                "scientific_execution": False,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
