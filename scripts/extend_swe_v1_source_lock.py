"""Extend V1 source pins from a byte-verified local inventory and upstream tree.

Does not fetch images, execute task code, or change existing pins.
"""

import argparse
import hashlib
import json
from pathlib import Path


def extend(lock, inventory, tree):
    if tree.get("truncated") or inventory["task_count"] != 731:
        raise ValueError("Complete frozen V1 inventory and Git tree required")
    if inventory["dataset_sha256"] != lock["dataset"]["parquet_sha256"]:
        raise ValueError("Dataset identity differs")
    if (
        inventory["source_receipt"]["official_v1"]["commit"]
        != lock["official_v1"]["commit"]
    ):
        raise ValueError("Upstream revision differs")
    blobs = {e["path"]: e for e in tree["tree"] if e["type"] == "blob"}
    for path, row in inventory["source_files"].items():
        if (
            path not in blobs
            or blobs[path]["sha"] != row["git_blob_sha1"]
            or blobs[path]["size"] != row["bytes"]
        ):
            raise ValueError("Source does not match upstream tree: " + path)
    expected = {}
    for key in ["sha256", "nodebb_task_source_sha256", "grading_dockerfiles_sha256"]:
        expected.update(lock["official_v1"].get(key, {}))
    for path, value in expected.items():
        if inventory["source_files"][path]["sha256"] != value:
            raise ValueError("Existing pin would change: " + path)
    identifiers = [t["instance_id"] for t in inventory["tasks"]]
    if len(identifiers) != 731 or len(set(identifiers)) != 731:
        raise ValueError("V1 task membership is incomplete or duplicated")
    result = json.loads(json.dumps(lock))
    # Retain the existing key/API; it historically covered only NodeBB.
    pins = result["official_v1"]["nodebb_task_source_sha256"]
    for task in inventory["tasks"]:
        identifier = task["instance_id"]
        required = [
            f"run_scripts/{identifier}/{name}"
            for name in ["run_script.sh", "parser.py"]
        ] + [
            f"dockerfiles/{kind}/{identifier}/Dockerfile"
            for kind in ["base_dockerfile", "instance_dockerfile"]
        ]
        if set(task["required_paths"]) != set(required):
            raise ValueError("Incorrect task source binding: " + identifier)
        for path in task["required_paths"]:
            pins[path] = inventory["source_files"][path]["sha256"]
    result["official_v1"]["task_source_scope"] = (
        "All731 V1 tasks across11 repositories; legacy nodebb_task_source_sha256 key retained"
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["lock", "inventory", "git-tree", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    result = extend(
        json.loads(a.lock.read_text()),
        json.loads(a.inventory.read_text()),
        json.loads(a.git_tree.read_text()),
    )
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            dict(
                output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                tasks=731,
            )
        )
    )


if __name__ == "__main__":
    main()
