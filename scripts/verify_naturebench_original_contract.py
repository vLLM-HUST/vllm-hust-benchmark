"""Verify pinned original source and canonical task groups without running agents.

No dataset acquisition, author module execution, model calls or scoring. Original
code/environment provenance is not a qualified scientific task runtime.
"""

import argparse
import ast
import json
from pathlib import Path

from prepare_jsonschemabench_author_source import verify_source

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/naturebench-author-source.json"
RELEASE = ROOT / "configs/workload-native/naturebench-original-release.json"


def task_groups(root, expected):
    groups = {}
    for name, count in expected.items():
        values = [
            line.strip()
            for line in (Path(root) / f"{name}.txt").read_text("utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if len(values) != count or len(set(values)) != count:
            raise ValueError("original task group count or uniqueness differs")
        groups[name] = values
    resource = groups["cpu"] + groups["gpu_high"] + groups["gpu_low"]
    if len(set(resource)) != len(resource) or set(resource) != set(groups["all"]):
        raise ValueError("original resource groups are not a disjoint full partition")
    return groups


def verify(root):
    root = verify_source(root, PINS)
    release = json.loads(RELEASE.read_text("utf-8"))
    pins = json.loads(PINS.read_text("utf-8"))
    if pins["revision"] != release["code_revision"]:
        raise ValueError("source and release revisions differ")
    full = task_groups(root / "task-set", release["full_task_groups"])
    small = task_groups(root / "task-set/naturebench-25", release["small_task_groups"])
    if full["cpu"] != release["official_cpu_tasks"] or any(
        not set(small[name]) <= set(full[name]) for name in full
    ):
        raise ValueError("original CPU list or small-track containment differs")
    python_files = [item for item in pins["files"] if item["path"].endswith(".py")]
    for item in python_files:
        ast.parse((root / item["path"]).read_text("utf-8"), filename=item["path"])
    return {
        "source_id": "NATUREBENCH",
        "status": "ORIGINAL_PROTOCOL_AND_CANONICAL_TASK_GROUPS_VERIFIED_NOT_NATIVE_DELIVERY",
        "code_revision": pins["revision"],
        "dataset_revision": release["dataset_revision"],
        "original_files_verified": len(pins["files"]),
        "original_bytes_verified": sum(item["bytes"] for item in pins["files"]),
        "python_files_syntax_checked_not_executed": len(python_files),
        "full_task_groups": {key: len(value) for key, value in full.items()},
        "small_task_groups": {key: len(value) for key, value in small.items()},
        "task_data_acquired_by_this_verifier": False,
        "task_data_verified": False,
        "model_calls": 0,
        "host_execution_of_task_or_generated_code": False,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.author_root)
    if args.output:
        if args.output.resolve().is_relative_to(args.author_root.resolve()):
            parser.error("evidence output must be separate from original source")
        args.output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
