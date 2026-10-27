"""Validate prepared originals and emit official commands for a qualified sandbox.

Does not execute any command. The paths in the plan are sandbox paths, not host
paths. A launcher must qualify isolation, dependency and resource support first.
"""

import argparse
import hashlib
import json
from pathlib import Path

from audit_bigcodebench_scoring_assets import verify_source
from run_bigcodebench_original_generation import CODE_REVISION, DATA_SHA, select_indices


def read_lines(path):
    return [
        json.loads(line)
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def build_plan(prepared, author):
    prepared = Path(prepared).resolve(strict=True)
    verified_source = verify_source(author)
    receipt = json.loads((prepared / "receipt-private.json").read_text("utf-8"))
    if (
        receipt.get("source_id") != "BIGCODEBENCH"
        or receipt.get("status") != "AUTHOR_SANITIZER_INPUT_PREPARED_NOT_EXECUTED"
        or receipt.get("code_revision") != CODE_REVISION
        or receipt.get("original_parquet_sha256_required_by_cli") != DATA_SHA
        or receipt.get("original_dataset_rows_preserved") != 1140
        or receipt.get("original_solution_text_modified") is not False
    ):
        raise ValueError(
            "unmodified original BigCodeBench preparation receipt required"
        )
    paths = {}
    for name in ("original-dataset-private.jsonl", "raw-solutions-private.jsonl"):
        path = (prepared / name).resolve(strict=True)
        if not path.is_relative_to(prepared):
            raise ValueError("prepared file escapes root")
        if hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get(
            "prepared_file_sha256", {}
        ).get(name):
            raise ValueError("prepared file checksum mismatch")
        paths[name] = path
    rows = read_lines(paths["original-dataset-private.jsonl"])
    solutions = read_lines(paths["raw-solutions-private.jsonl"])
    if [r.get("task_id") for r in rows] != [f"BigCodeBench/{i}" for i in range(1140)]:
        raise ValueError("full original ordered dataset required")
    selected_ids = receipt.get("selected_task_ids")
    if not isinstance(selected_ids, list):
        raise ValueError("original task IDs required")
    try:
        indices = [int(task.removeprefix("BigCodeBench/")) for task in selected_ids]
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("invalid original task IDs") from error
    select_indices(1140, indices=indices)
    if selected_ids != [f"BigCodeBench/{i}" for i in indices]:
        raise ValueError("noncanonical original IDs")
    if (
        "selected_original_row_indices" in receipt
        and receipt["selected_original_row_indices"] != indices
    ):
        raise ValueError("original selection identity mismatch")
    if [s.get("task_id") for s in solutions] != selected_ids or any(
        not isinstance(s.get("solution"), str) or "completion" in s for s in solutions
    ):
        raise ValueError("original full solution selection mismatch")
    sanitized = "/output/raw-solutions-private-sanitized.jsonl"
    return {
        "source_id": "BIGCODEBENCH",
        "status": "OFFICIAL_SANDBOX_COMMAND_PLAN_PREPARED_NOT_EXECUTED",
        "code_revision": CODE_REVISION,
        "publisher_source_checksums": verified_source,
        "prepared_file_sha256": {
            name: receipt["prepared_file_sha256"][name] for name in paths
        },
        "selected_original_row_indices": indices,
        "selected_task_ids": selected_ids,
        "validation_scope": {
            "original_dataset_rows_preserved": 1140,
            "selected_unique_tasks": len(selected_ids),
            "all_original_tasks_selected": len(selected_ids) == 1140,
            "sampling_does_not_remove_native_steps": True,
            "required_per_selected_task": [
                "author sanitize",
                "author canonical solution/environment checks",
                "author candidate functional evaluation",
            ],
            "boundary": "A complete sampled native workflow is not a full-suite quality result. Dataset preservation does not imply all tasks executed.",
        },
        "required_environment": {
            "BIGCODEBENCH_OVERRIDE_PATH": "/inputs/original-dataset-private.jsonl",
            "XDG_CACHE_HOME": "/output/fresh-cache",
            "PYTHONPATH": "/author",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
        "required_staging": {
            "/inputs/original-dataset-private.jsonl": "read-only original full prepared dataset",
            "/output/raw-solutions-private.jsonl": "byte-identical prepared solutions copied into fresh private output",
            "/author": "read-only complete fixed official source; all imports available",
        },
        "commands": [
            [
                "/opt/python/bin/python3",
                "-m",
                "bigcodebench.sanitize",
                "--samples",
                "/output/raw-solutions-private.jsonl",
            ],
            [
                "/opt/python/bin/python3",
                "-m",
                "bigcodebench.evaluate",
                "--execution",
                "local",
                "--split",
                "instruct",
                "--subset",
                "full",
                "--samples",
                sanitized,
                "--selective_evaluate",
                ",".join(selected_ids),
                "--pass_k",
                "1",
                "--calibrated",
                "True",
                "--no_gt",
                "False",
                "--min_time_limit",
                "1",
                "--max_as_limit",
                "30720",
                "--max_data_limit",
                "30720",
                "--max_stack_limit",
                "10",
            ],
        ],
        "required_controls": [
            "Qualified private user/mount/network/PID namespaces, no host data, credentials or NPU devices; no host fallback.",
            "Fresh writable output/cache; retain all failures. Canonical-solution checks must pass for every selected task before accepting model scores.",
            "Author 30GiB address/data limits and 10MiB stack limits must be supported; DS1000's 2GiB hard address limit is not a compatible substitute.",
            "Preserve official sanitizer and evaluator defaults including worker choice; qualify concurrency instead of silently reducing it.",
            "Verify dependency imports, runtime data files, sanitized selected IDs and exact official result coverage. Missing dependencies are environment failure, not wrong model answers.",
        ],
        "functional_scoring_executed": False,
        "environment_qualified": False,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--official-code", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_plan(args.prepared_dir, args.official_code), indent=2))


if __name__ == "__main__":
    main()
