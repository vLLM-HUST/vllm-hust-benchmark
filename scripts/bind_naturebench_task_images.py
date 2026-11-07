"""Bind qualified image IDs to original task tags; never replace an existing image.

The manifest declares qualification externally; binding does not prove it or run
an Agent. Default is an offline plan. No Docker socket goes into task containers.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TASKS = json.loads(
    (ROOT / "configs/workload-native/naturebench-original-release.json").read_text(
        "utf-8"
    )
)["official_cpu_tasks"]


def bindings(manifest):
    if not isinstance(manifest, dict) or not manifest or set(manifest) - set(TASKS):
        raise ValueError("distinct registered original CPU task IDs required")
    rows = []
    for task, identity in sorted(manifest.items()):
        if not isinstance(identity, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", identity
        ):
            raise ValueError("full immutable Docker image ID required, not a tag")
        rows.append(
            {
                "task_id": task,
                "image_id": identity,
                "tag": f"naturebench-task-{task}:base",
            }
        )
    return rows


def bind(manifest, *, execute=False, call=subprocess.run):
    rows = bindings(manifest)
    if execute:
        # Check every source and destination before creating any alias.
        for row in rows:
            source = call(
                ["docker", "image", "inspect", row["image_id"]],
                capture_output=True,
                text=True,
                check=True,
            )
            if json.loads(source.stdout)[0]["Id"] != row["image_id"]:
                raise ValueError("source image identity differs")
            listed = call(
                ["docker", "image", "ls", "--no-trunc", "--quiet", row["tag"]],
                capture_output=True,
                text=True,
                check=True,
            )
            existing = set(listed.stdout.split())
            if existing and existing != {row["image_id"]}:
                raise ValueError(
                    "original task tag already refers to another image; no overwrite"
                )
        for row in rows:
            # Recheck the destination immediately before the local alias write.
            listed = call(
                ["docker", "image", "ls", "--no-trunc", "--quiet", row["tag"]],
                capture_output=True,
                text=True,
                check=True,
            )
            if set(listed.stdout.split()) - {row["image_id"]}:
                raise ValueError("task tag changed during preflight; no overwrite")
            call(["docker", "image", "tag", row["image_id"], row["tag"]], check=True)
            actual = call(
                ["docker", "image", "inspect", row["tag"]],
                capture_output=True,
                text=True,
                check=True,
            )
            if json.loads(actual.stdout)[0]["Id"] != row["image_id"]:
                raise ValueError("bound tag identity differs")
    return {
        "source_id": "NATUREBENCH",
        "status": (
            "IMAGE_ALIASES_BOUND_NOT_NATIVE_EXECUTION"
            if execute
            else "OFFLINE_IMAGE_BINDING_PLAN"
        ),
        "bindings": rows,
        "qualification_independently_verified": False,
        "native_tasks": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="JSON object mapping original task ID to qualified sha256 Docker image ID",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            bind(json.loads(args.manifest.read_text("utf-8")), execute=args.execute),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
