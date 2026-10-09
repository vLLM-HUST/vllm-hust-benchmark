"""Audit frozen SWE-bench-Pro task/image provenance without executing tasks.

Python 3.11+ and PyArrow are required. Root-license identification is provenance,
not blanket clearance of derived task contents or bundled third-party assets.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

DATA_REVISION = "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"  # pragma: allowlist secret
DATA_SHA256 = "1067f72c1f43bbc578276888625db3b836cebb535546c9eaf824c5d5dd104a7f"  # pragma: allowlist secret
HARNESS_REVISION = (
    "66f92766bba642462d4bbe5479e83f91f9211862"  # pragma: allowlist secret
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def checked(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if sha(data) != digest.removeprefix("sha256:"):
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    return data


def check_blob(data: bytes, digest: str) -> None:
    if (
        hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        != digest
    ):
        raise ValueError("Git blob mismatch")


def test_list(value: object, *, allow_literal: bool = False) -> list[str]:
    result = value
    if isinstance(value, str):
        try:
            result = json.loads(value)
        except json.JSONDecodeError:
            if not allow_literal:
                raise
            # Some pinned harness configs retain Python list literals. The HF
            # release uses JSON; accept only equivalent literal string lists.
            result = ast.literal_eval(value)
    if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
        raise ValueError("Expected list of test identifiers")
    return result


def audit_task(row: dict, receipt: dict, assets: Path, tree: dict) -> dict:
    import tomllib

    task_id = row["instance_id"]
    if row["version"] != "2.0.0":
        raise ValueError("Unexpected task release")
    for key in ("instance_id", "repo", "base_commit", "docker_image"):
        if receipt[key] != row[key]:
            raise ValueError(f"Task receipt differs: {key}")
    if receipt["harness_revision"] != HARNESS_REVISION:
        raise ValueError("Harness revision differs")
    if row["docker_image"] != f"ghcr.io/scaleapi/swe-bench_pro-v2:{task_id}":
        raise ValueError("Unexpected image repository or tag")
    # Confirm every declared file against the pinned upstream tree. For files
    # not downloaded, this proves identity/existence only, not execution.
    required_files = {
        "task.toml",
        "environment/Dockerfile",
        "tests/config.json",
        "tests/test.sh",
        "tests/run_script.sh",
        "tests/parser.py",
        "tests/test_patch.patch",
        "solution/gold_patch.diff",
        "instruction.md",
    }
    if not required_files.issubset(receipt["harness_files"]):
        raise ValueError("Missing required harness file")
    for name, metadata in receipt["harness_files"].items():
        path = f"v2/tasks/{task_id}/{name}"
        if tree[path]["sha"] != metadata["git_blob_sha"]:
            raise ValueError("Harness tree/blob mismatch")
    metadata = {}
    for name in ("task.toml", "environment/Dockerfile", "tests/config.json"):
        entry = receipt["harness_files"][name]
        data = checked(
            assets / "harness-metadata" / task_id / name.replace("/", "-"),
            entry["sha256"],
        )
        check_blob(data, entry["git_blob_sha"])
        metadata[name] = data
    task = tomllib.loads(metadata["task.toml"].decode())
    if task["task"]["name"] != "swebench-pro/" + task_id:
        raise ValueError("Harbor task identity differs")
    if task["environment"]["docker_image"] != row["docker_image"]:
        raise ValueError("Harbor image mapping differs")
    dockerfile_lines = metadata["environment/Dockerfile"].decode().strip().splitlines()
    if dockerfile_lines != ["FROM " + row["docker_image"]]:
        raise ValueError("Unexpected Dockerfile modifications")
    tests = json.loads(metadata["tests/config.json"])
    for key in ("fail_to_pass", "pass_to_pass", "selected_test_files_to_run"):
        if test_list(tests[key], allow_literal=True) != test_list(row[key]):
            raise ValueError(f"Verifier task data differs: {key}")
    license = receipt["root_license"]
    license_bytes = checked(
        assets / "license-texts" / (license["sha256"] + ".txt"), license["sha256"]
    )
    check_blob(license_bytes, license["sha"])
    if f"/blob/{row['base_commit']}/" not in license["html_url"]:
        raise ValueError("License not pinned to task base commit")
    gold_patch = row["patch"].encode()
    check_blob(
        gold_patch, receipt["harness_files"]["solution/gold_patch.diff"]["git_blob_sha"]
    )
    dataset_test_patch = row["test_patch"].encode()
    expected_test_blob = receipt["harness_files"]["tests/test_patch.patch"][
        "git_blob_sha"
    ]
    dataset_test_blob = hashlib.sha1(
        b"blob " + str(len(dataset_test_patch)).encode() + b"\0" + dataset_test_patch
    ).hexdigest()
    test_patch_exact = dataset_test_blob == expected_test_blob
    harness_test_patch = dataset_test_patch
    if not test_patch_exact:
        harness_test_patch = (
            assets / "harness-metadata" / task_id / "tests-test_patch.patch"
        ).read_bytes()
        check_blob(harness_test_patch, expected_test_blob)
        if harness_test_patch.replace(b"\r\n", b"\n") != dataset_test_patch:
            raise ValueError("Unexplained dataset/harness test patch difference")
    # Verifier fixture line endings can be semantically significant. Keep the
    # pinned harness bytes; never reconstruct this patch from normalized HF text.
    image = receipt["image"]
    if "index" in image["media_type"] or "manifest.list" in image["media_type"]:
        resolution = json.loads(
            (assets / "index-resolutions" / (task_id + ".json")).read_text()
        )
        if resolution.get("error") or resolution["index_digest"] != image["digest"]:
            raise ValueError("Image index unresolved")
        folder = assets / "index-resolutions" / task_id
        index = json.loads(checked(folder / "index.json", image["digest"]))
        matches = [
            m
            for m in index["manifests"]
            if m.get("platform", {}).get("os") == "linux"
            and m["platform"].get("architecture") == "amd64"
        ]
        if len(matches) != 1 or matches[0]["digest"] != resolution["manifest_digest"]:
            raise ValueError("Ambiguous or changed image platform")
        manifest_digest = resolution["manifest_digest"]
        manifest = json.loads(checked(folder / "manifest.json", manifest_digest))
        config_path = folder / "config.json"
    else:
        manifest_digest = image["digest"]
        manifest = json.loads(
            checked(assets / "image-manifests" / (task_id + ".json"), manifest_digest)
        )
        config_path = assets / "image-configs" / (task_id + ".json")
    config_digest = manifest["config"]["digest"]
    config = json.loads(checked(config_path, config_digest))
    if (config["os"], config["architecture"]) != ("linux", "amd64"):
        raise ValueError("Expected linux/amd64 task image")
    return {
        "instance_id": task_id,
        "repo": row["repo"],
        "base_commit": row["base_commit"],
        "release": row["version"],
        "hard": row["hard"],
        "source_row_sha256": sha(canonical(row)),
        "problem_sha256": sha(row["problem_statement"].encode()),
        "patch_sha256": sha(row["patch"].encode()),
        "test_patch_sha256": sha(dataset_test_patch),
        "harness_test_patch_sha256": sha(harness_test_patch),
        "test_patch_byte_identical": test_patch_exact,
        "test_patch_difference": None
        if test_patch_exact
        else "CRLF preserved in harness; LF in HF data",
        "verifier_patch_authority": "pinned harness bytes",
        "image_tag": row["docker_image"],
        "tag_resolved_digest": image["digest"],
        "linux_amd64_manifest_digest": manifest_digest,
        "image_config_digest": config_digest,
        "compressed_layer_bytes": sum(x["size"] for x in manifest["layers"]),
        "image_layers": {x["digest"]: x["size"] for x in manifest["layers"]},
        "root_license": license,
        "harness_files": receipt["harness_files"],
        "test_counts": {
            key: len(test_list(row[key]))
            for key in ("fail_to_pass", "pass_to_pass", "selected_test_files_to_run")
        },
        "environment": task["environment"],
        "agent": task["agent"],
        "verifier": task["verifier"],
        "retrieval_receipt_sha256": sha(canonical(receipt)),
        "runtime_repo_head_verified": False,
        "tests_executed": False,
    }


def audit(args) -> None:
    import pyarrow.parquet as pq

    checked(args.data, DATA_SHA256)
    rows = pq.read_table(args.data).to_pylist()
    if len(rows) != 642 or len({row["instance_id"] for row in rows}) != 642:
        raise ValueError("Expected 642 unique V2 tasks")
    source_tree = json.loads(args.tree.read_text())
    if source_tree["revision"] != HARNESS_REVISION or source_tree["tree"]["truncated"]:
        raise ValueError("Incomplete or unexpected harness tree")
    tree = {entry["path"]: entry for entry in source_tree["tree"]["tree"]}
    inventory, failures = [], []
    for row in rows:
        try:
            receipt = json.loads(
                (args.assets / (row["instance_id"] + ".json")).read_text()
            )
            inventory.append(audit_task(row, receipt, args.assets, tree))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(
                {
                    "instance_id": row["instance_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "task-inventory.jsonl").write_text(
        "".join(canonical(r).decode() + "\n" for r in inventory)
    )
    layers = {}
    for row in inventory:
        for digest, size in row["image_layers"].items():
            if digest in layers and layers[digest] != size:
                raise ValueError("Inconsistent layer size for one digest")
            layers[digest] = size
    summary = {
        "scope": "input/image/license provenance only; no sandbox execution or resolved-rate score",
        "dataset_revision": DATA_REVISION,
        "data_sha256": DATA_SHA256,
        "harness_revision": HARNESS_REVISION,
        "harness_tree_sha256": sha(args.tree.read_bytes()),
        "test_patch_byte_differences": [
            row["instance_id"]
            for row in inventory
            if not row["test_patch_byte_identical"]
        ],
        "rows": len(rows),
        "verified": len(inventory),
        "failures": failures,
        "repositories": dict(Counter(row["repo"] for row in inventory)),
        "root_license_identifications": dict(
            Counter(row["root_license"]["license"]["spdx_id"] for row in inventory)
        ),
        "unique_compressed_layers_bytes": sum(layers.values()),
        "unique_layers": len(layers),
        "image_layers_downloaded": False,
        "runtime_repo_heads_verified": False,
        "rights_clearance": "not inferred from root-license identification",
        "remaining": [
            "Supported linux/amd64 isolated sandbox backend",
            "Agent/tool/budget contract and fresh-sandbox regrading",
            "Task and bundled third-party rights review",
            "Real task trajectories and compression applicability audit",
        ],
        "scores": None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "rows",
                    "verified",
                    "repositories",
                    "root_license_identifications",
                )
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(f"Incomplete audit: {len(failures)} failures retained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "tree", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    audit(parser.parse_args())
