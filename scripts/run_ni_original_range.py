"""Run fixed original NI ranges/full files using the unchanged validated core.

Only original-instance selection changes. Official collator/scorer, full input,
generation budget and response integrity remain in the pinned existing core.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from run_ni_editing_extraction_original import load_core, TASKS as EXTRA_TASKS


def selected_indices(total, start=None, count=None, full=False):
    if full:
        if start is not None or count is not None:
            raise ValueError("--full cannot be combined with --start/--count")
        return list(range(total))
    start = 0 if start is None else start
    count = 3 if count is None else count
    if start < 0 or count < 1 or start + count > total:
        raise ValueError("range must be within the original task file")
    return list(range(start, start + count))


def main():
    core = load_core()
    tasks = {**core.TASKS, **EXTRA_TASKS}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-files", nargs="+", choices=tuple(tasks), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--collator-source", type=Path, required=True)
    parser.add_argument("--scorer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--start", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if len(set(args.task_files)) != len(args.task_files):
        parser.error("select distinct original task files")
    try:
        selections = {
            name: selected_indices(tasks[name][1], args.start, args.count, args.full)
            for name in args.task_files
        }
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    if output == args.data_root.resolve() or args.data_root.resolve() in output.parents:
        parser.error("output must be outside original data")
    sidecar = output.with_name(output.name + ".selection.private.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation freezes task IDs before any core/model execution.
    with sidecar.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "selected_original_indices": selections,
                "full": args.full,
                "driver_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "core_sha256": hashlib.sha256(
                    Path(core.__file__).read_bytes()
                ).hexdigest(),
            },
            stream,
            indent=2,
        )
    sidecar.chmod(0o600)
    original_tasks, original_selector, original_argv = (
        core.TASKS,
        core.selected_indices,
        sys.argv,
    )
    try:
        core.TASKS = tasks
        core.selected_indices = lambda name, full_small: selections[name]
        sys.argv = [
            core.__file__,
            "--task-files",
            *args.task_files,
            "--data-root",
            str(args.data_root),
            "--collator-source",
            str(args.collator_source),
            "--scorer-root",
            str(args.scorer_root),
            "--output",
            str(output),
            "--endpoint",
            args.endpoint,
            "--model",
            args.model,
        ]
        core.main()
        receipt = json.loads((output / "receipt.json").read_text("utf-8"))
        for name, indices in selections.items():
            actual = [
                sample["index"]
                for sample in receipt["samples"]
                if sample["task_file"] == name
            ]
            if actual != indices:
                raise ValueError(
                    "actual original task IDs differ from frozen selection"
                )
        # Preserve core receipt bytes; its legacy 'first_three' label is not
        # authoritative for this adapter. Sidecar + actual IDs define scope.
        result = {
            "status": "ORIGINAL_RANGE_EXECUTED_AND_SCORED",
            "selected_original_indices": selections,
            "core_receipt_sha256": hashlib.sha256(
                (output / "receipt.json").read_bytes()
            ).hexdigest(),
            "legacy_core_selection_label": receipt["selection"],
            "selection_authority": sidecar.name,
            "performance_claim": False,
        }
        target = output / "range-receipt.json"
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        target.chmod(0o600)
    finally:
        core.TASKS, core.selected_indices, sys.argv = (
            original_tasks,
            original_selector,
            original_argv,
        )


if __name__ == "__main__":
    main()
