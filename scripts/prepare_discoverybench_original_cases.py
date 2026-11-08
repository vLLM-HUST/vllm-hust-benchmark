"""Prepare original selected scientific cases outside the repository; no models.

Metadata, scientific data and scoring references are separate so a tool container
can mount only data. This does not start an Agent, Docker, a model or a scorer.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil


def selected_records(records, positions):
    if (
        not positions
        or len(set(positions)) != len(positions)
        or any(p < 0 or p >= len(records) for p in positions)
    ):
        raise ValueError("explicit distinct original positions in range required")
    selected = []
    for pos in positions:
        record = records[pos]
        if record.runtime["blockers"]:
            raise ValueError(
                f"original task {record.task['task_id']} blocked; do not replace it: {record.runtime['blockers']}"
            )
        selected.append((pos, record))
    return selected


def output_path(data_root, output):
    data_root, output = Path(data_root).resolve(strict=True), Path(output).resolve()
    repository = Path(__file__).resolve().parents[1]
    if output.exists():
        raise FileExistsError(
            "fresh external run directory required; preserve previous evidence"
        )
    if (
        output.is_relative_to(repository)
        or output.is_relative_to(data_root)
        or data_root.is_relative_to(output)
    ):
        raise ValueError(
            "private output must be outside repository and original data root"
        )
    return data_root, output


def prepare(args):
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.assets import asset_path, verify_file
    from vllm_hust_workload_data.discoverybench import restore_original_case

    data_root, output = output_path(args.data_root, args.output)
    records = list(iter_workload("DISCOVERYBENCH", data_root=data_root))
    selected = selected_records(records, args.positions)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    cases = []
    for pos, record in selected:
        native = restore_original_case(record, data_root=data_root)
        case = output / f"case-{pos}"
        (case / "metadata").mkdir(parents=True)
        (case / "data").mkdir()
        metadata_name = record.runtime["original_metadata_path"]
        files = {item["path"]: item for item in record.runtime["native_files"]}
        metadata_source = asset_path(data_root, metadata_name)
        metadata_target = case / "metadata" / metadata_source.name
        shutil.copyfile(metadata_source, metadata_target)
        if (
            hashlib.sha256(metadata_target.read_bytes()).hexdigest()
            != files[metadata_name]["sha256"]
        ):
            raise ValueError("copied original metadata differs")
        data_files = []
        for item in native["metadata"]["datasets"]:
            relative = PurePosixPath(item["name"])
            target = case / "data" / str(relative)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not target.resolve().is_relative_to(case / "data")
            ):
                raise ValueError(
                    "original scientific data path escapes isolated data directory"
                )
            name = str(PurePosixPath(metadata_name).parent / relative)
            verify_file(data_root, files[name])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(asset_path(data_root, name), target)
            if hashlib.sha256(target.read_bytes()).hexdigest() != files[name]["sha256"]:
                raise ValueError("copied original scientific data differs")
            data_files.append(
                {
                    "path": str(target.relative_to(output)),
                    "sha256": files[name]["sha256"],
                }
            )
        reference = case / "reference.json"
        reference.write_text(
            json.dumps(native["answer_key_row"], ensure_ascii=False), encoding="utf-8"
        )
        cases.append(
            {
                "original_position": pos,
                "original_task_id": record.task["task_id"],
                "query_id": native["answer_key_row"]["query_id"],
                "metadata": {
                    "path": str(metadata_target.relative_to(output)),
                    "sha256": hashlib.sha256(metadata_target.read_bytes()).hexdigest(),
                },
                "reference": {
                    "path": str(reference.relative_to(output)),
                    "sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                },
                "data_files": data_files,
            }
        )
    manifest = {
        "schema_version": "discoverybench-original-cases/v1",
        "source_id": "DISCOVERYBENCH",
        "status": "ORIGINAL_CASES_PREPARED_NOT_EXECUTED",
        "cases": cases,
        "original_actor_iterations": 5,
        "original_python_call_limit": 1,
        "tool_mount_policy": "Mount only case data, never metadata/main hypotheses, reference, model queue or Docker socket",
        "new_model_calls": 0,
        "source_checkbox_eligible": False,
    }
    (output / "cases-private.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "original_positions": args.positions,
                "model_calls": 0,
            }
        )
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--positions", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {"status": "PLAN_ONLY", "source_id": "DISCOVERYBENCH", "model_calls": 0}
            )
        )
        return 0
    if args.data_root is None or args.output is None:
        parser.error("explicit external data root and fresh private output required")
    os.umask(0o077)
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
