"""Verify pinned Terminal-Bench 2.1 sources and image metadata offline.

This audit neither pulls image layers nor runs tasks. Python 3.11+ is required.
Root repository licensing does not establish rights to every bundled asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath

REVISION = "7131e4375048a0e408a8fb404b5f499d726b695b"  # pragma: allowlist secret
TREE_SHA256 = "c7cf0a039f77fa21292e1436f4a22293c2179e26532b4838c289ff6b1f9c26f9"  # pragma: allowlist secret
EXPECTED_TASKS = 89


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if sha(data) != digest.removeprefix("sha256:"):
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    return data


def check_blob(data: bytes, digest: str) -> None:
    actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if actual != digest:
        raise ValueError("Git blob mismatch")


def verify_snapshot(snapshot: Path, tree: dict) -> dict[str, dict]:
    if tree["sha"] != REVISION or tree.get("truncated") is not False:
        raise ValueError("Wrong revision or incomplete Git tree")
    entries = {}
    for entry in tree["tree"]:
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or ".." in path.parts or str(path) != entry["path"]:
            raise ValueError("Unsafe Git tree path")
        if entry["type"] == "tree":
            continue
        if entry["type"] != "blob" or entry["mode"] not in {"100644", "100755"}:
            raise ValueError("Unsupported Git tree entry")
        if str(path) in entries:
            raise ValueError("Duplicate Git tree path")
        local = snapshot.joinpath(*path.parts)
        if local.is_symlink() or any(
            p.is_symlink() for p in local.parents if p != snapshot.parent
        ):
            raise ValueError("Symlink in snapshot path")
        data = local.read_bytes()
        check_blob(data, entry["sha"])
        if len(data) != entry["size"]:
            raise ValueError("Git blob size mismatch")
        entries[str(path)] = {
            "git_blob_sha": entry["sha"],
            "sha256": sha(data),
            "bytes": len(data),
        }
    actual_files = {
        str(p.relative_to(snapshot)) for p in snapshot.rglob("*") if p.is_file()
    }
    if actual_files != set(entries):
        raise ValueError("Snapshot has missing or extra files")
    return entries


def audit_image(folder: Path, receipt: dict) -> dict:
    if receipt.get("error"):
        raise ValueError("Image retrieval failed")
    tagged = json.loads(checked(folder / "tag-manifest.json", receipt["tag_digest"]))
    manifest_bytes = checked(
        folder / "platform-manifest.json", receipt["manifest_digest"]
    )
    manifest = json.loads(manifest_bytes)
    if "manifests" in tagged:
        matches = [
            m
            for m in tagged["manifests"]
            if m.get("platform", {}).get("os") == "linux"
            and m["platform"].get("architecture") == "amd64"
        ]
        if len(matches) != 1 or matches[0]["digest"] != receipt["manifest_digest"]:
            raise ValueError("Image platform is missing or ambiguous")
        if matches[0]["size"] != len(manifest_bytes):
            raise ValueError("Image descriptor size mismatch")
    elif receipt["tag_digest"] != receipt["manifest_digest"]:
        raise ValueError("Image tag does not bind platform manifest")
    if manifest.get("schemaVersion") != 2:
        raise ValueError("Unsupported image manifest")
    config = manifest["config"]
    if config["digest"] != receipt["config_digest"]:
        raise ValueError("Image configuration descriptor mismatch")
    config_bytes = checked(folder / "image-config.json", config["digest"])
    if config["size"] != len(config_bytes):
        raise ValueError("Image configuration size mismatch")
    identity = json.loads(config_bytes)
    if (identity["os"], identity["architecture"]) != ("linux", "amd64"):
        raise ValueError("Image architecture differs from frozen execution requirement")
    if (receipt["os"], receipt["architecture"]) != (
        identity["os"],
        identity["architecture"],
    ):
        raise ValueError("Image architecture receipt mismatch")
    layers = {}
    compressed_bytes = 0
    for layer in manifest["layers"]:
        digest, size = layer["digest"], layer["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("Invalid image layer size")
        if digest in layers and layers[digest] != size:
            raise ValueError("Conflicting image layer size")
        layers[digest] = size
        compressed_bytes += size
    if (
        layers != receipt["layers"]
        or compressed_bytes != receipt["compressed_layer_bytes"]
    ):
        raise ValueError("Image layer receipt mismatch")
    return {
        "tag_digest": receipt["tag_digest"],
        "manifest_digest": receipt["manifest_digest"],
        "config_digest": config["digest"],
        "platform": "linux/amd64",
        "compressed_layer_bytes": compressed_bytes,
        "layers": layers,
    }


def audit_task(task_id: str, assets: Path, files: dict) -> dict:
    import tomllib

    base = "tasks/" + task_id + "/"
    task_files = {
        name.removeprefix(base): metadata
        for name, metadata in files.items()
        if name.startswith(base)
    }
    required = {
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "tests/test.sh",
        "solution/solve.sh",
    }
    if not required.issubset(task_files):
        raise ValueError("Required task file is missing")
    task = tomllib.loads((assets / "snapshot" / base / "task.toml").read_text())
    if (
        task["schema_version"] != "1.1"
        or task["task"]["name"] != "terminal-bench/" + task_id
    ):
        raise ValueError("Task identity or schema mismatch")
    receipt = json.loads((assets / "image-audit" / (task_id + ".json")).read_text())
    if (
        receipt["task_id"] != task_id
        or receipt["task_toml_sha256"] != task_files["task.toml"]["sha256"]
    ):
        raise ValueError("Task receipt identity mismatch")
    for key in ("environment", "agent", "verifier"):
        if receipt[key] != task[key]:
            raise ValueError(f"Task receipt contract differs: {key}")
    image = task["environment"]["docker_image"]
    if image != receipt["image"]:
        raise ValueError("Task image mapping differs")
    image_audit = audit_image(assets / "image-audit" / task_id, receipt)
    environment = task["environment"]
    # Publish resource requirements and hashes, never image environment values.
    resources = {
        key: environment[key]
        for key in (
            "cpus",
            "memory_mb",
            "storage_mb",
            "gpus",
            "build_timeout_sec",
            "allow_internet",
        )
    }
    return {
        "task_id": task_id,
        "task_name": task["task"]["name"],
        "source_revision": REVISION,
        "files": task_files,
        "image": image,
        "image_audit": image_audit,
        "resources": resources,
        "agent_timeout_sec": task["agent"]["timeout_sec"],
        "verifier_timeout_sec": task["verifier"]["timeout_sec"],
        "mcp_server_count": len(environment["mcp_servers"]),
        "rights_review_complete": False,
        "image_layers_pulled": False,
        "task_executed": False,
    }


def audit(assets: Path, output: Path) -> dict:
    tree = json.loads(checked(assets / "tree.json", TREE_SHA256))
    files = verify_snapshot(assets / "snapshot", tree)
    task_ids = sorted(
        name.split("/")[1]
        for name in files
        if name.startswith("tasks/")
        and name.count("/") == 2
        and name.endswith("/task.toml")
    )
    if len(task_ids) != EXPECTED_TASKS or len(set(task_ids)) != EXPECTED_TASKS:
        raise ValueError("Frozen task inventory count differs")
    rows = [audit_task(task_id, assets, files) for task_id in task_ids]
    layers = {}
    for row in rows:
        for digest, size in row["image_audit"]["layers"].items():
            if digest in layers and layers[digest] != size:
                raise ValueError("Cross-task layer size differs")
            layers[digest] = size
    summary = {
        "scope": "source and image metadata audit; no task execution or score",
        "source_repo": "harbor-framework/terminal-bench-2-1",
        "source_revision": REVISION,
        "tasks": len(rows),
        "verified_source_files": len(files),
        "root_license": files["LICENSE"],
        "platforms": dict(Counter(r["image_audit"]["platform"] for r in rows)),
        "unique_compressed_layers": len(layers),
        "unique_compressed_layer_bytes": sum(layers.values()),
        "largest_image_compressed_bytes": max(
            r["image_audit"]["compressed_layer_bytes"] for r in rows
        ),
        "resource_values": {
            k: sorted({r["resources"][k] for r in rows}) for k in rows[0]["resources"]
        },
        "agent_timeout_values_sec": sorted({r["agent_timeout_sec"] for r in rows}),
        "verifier_timeout_values_sec": sorted(
            {r["verifier_timeout_sec"] for r in rows}
        ),
        "rights_review_complete": False,
        "image_layers_pulled": False,
        "tasks_executed": False,
        "remaining": [
            "Task and bundled third-party rights review",
            "Supported AMD64 fresh-sandbox backend",
            "Pinned Harbor and agent versions; model API connectivity",
            "Tool/context/token/retry budgets and verifier execution contract",
            "Fresh-sandbox verifier validation and matched B0/B1 task runs",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "inventory.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.assets, args.output), indent=2))
