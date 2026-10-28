"""Verify publisher CPU package relations without Agent or scientific execution."""

import argparse
import json

from vllm_hust_workload_data.naturebench import original_cpu_packages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
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
    print(
        json.dumps(
            {
                "status": "ORIGINAL_CPU_PACKAGE_RELATIONS_VERIFIED_NOT_NATIVE_EXECUTION",
                "packages": packages,
                "model_calls": 0,
                "scientific_execution": False,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
