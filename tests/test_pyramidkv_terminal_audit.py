"""Synthetic integrity fixtures, not sandbox execution or benchmark scores."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "terminal_audit", Path(__file__).parents[1] / "scripts/audit_pyramidkv_terminal.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def snapshot_fixture(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    data = b"synthetic task"
    (snapshot / "instruction.md").write_bytes(data)
    tree = {
        "sha": audit.REVISION,
        "truncated": False,
        "tree": [
            {
                "path": "instruction.md",
                "type": "blob",
                "mode": "100644",
                "size": len(data),
                "sha": hashlib.sha1(
                    b"blob " + str(len(data)).encode() + b"\0" + data
                ).hexdigest(),
            }
        ],
    }
    return snapshot, tree


def test_snapshot_exact(tmp_path):
    snapshot, tree = snapshot_fixture(tmp_path)
    assert audit.verify_snapshot(snapshot, tree)["instruction.md"]["bytes"] == 14


@pytest.mark.parametrize(
    "change",
    [
        "revision",
        "truncated",
        "content",
        "extra",
        "traversal",
        "symlink",
        "size",
        "duplicate",
    ],
)
def test_snapshot_rejects_tampering(tmp_path, change):
    snapshot, tree = snapshot_fixture(tmp_path)
    if change == "revision":
        tree["sha"] = "b" * 40
    elif change == "truncated":
        tree["truncated"] = True
    elif change == "content":
        (snapshot / "instruction.md").write_text("changed")
    elif change == "extra":
        (snapshot / "extra").write_text("extra")
    elif change == "traversal":
        tree["tree"][0]["path"] = "../instruction.md"
    elif change == "symlink":
        (snapshot / "instruction.md").rename(tmp_path / "outside")
        (snapshot / "instruction.md").symlink_to(tmp_path / "outside")
    elif change == "size":
        tree["tree"][0]["size"] += 1
    elif change == "duplicate":
        tree["tree"].append(tree["tree"][0])
    with pytest.raises(ValueError):
        audit.verify_snapshot(snapshot, tree)


def image_fixture(tmp_path, *, indexed=False, architecture="amd64", ambiguous=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = json.dumps({"os": "linux", "architecture": architecture}).encode()
    config_digest = "sha256:" + audit.sha(config)
    layer = {"digest": "sha256:" + "a" * 64, "size": 123}
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "config": {"digest": config_digest, "size": len(config)},
            "layers": [layer],
        }
    ).encode()
    manifest_digest = "sha256:" + audit.sha(manifest)
    if indexed:
        descriptor = {
            "digest": manifest_digest,
            "size": len(manifest),
            "platform": {"os": "linux", "architecture": "amd64"},
        }
        tag = json.dumps({"manifests": [descriptor] * (2 if ambiguous else 1)}).encode()
    else:
        tag = manifest
    (tmp_path / "tag-manifest.json").write_bytes(tag)
    (tmp_path / "platform-manifest.json").write_bytes(manifest)
    (tmp_path / "image-config.json").write_bytes(config)
    return {
        "tag_digest": "sha256:" + audit.sha(tag),
        "manifest_digest": manifest_digest,
        "config_digest": config_digest,
        "os": "linux",
        "architecture": architecture,
        "layers": {layer["digest"]: 123},
        "compressed_layer_bytes": 123,
    }


@pytest.mark.parametrize("indexed", [False, True])
def test_image_digest_chain(tmp_path, indexed):
    receipt = image_fixture(tmp_path, indexed=indexed)
    result = audit.audit_image(tmp_path, receipt)
    assert result["platform"] == "linux/amd64"
    assert result["compressed_layer_bytes"] == 123


@pytest.mark.parametrize(
    "change",
    [
        "config_bytes",
        "manifest_bytes",
        "tag_bytes",
        "config_digest",
        "architecture",
        "layers",
        "total",
        "ambiguous",
    ],
)
def test_image_rejects_tampering(tmp_path, change):
    receipt = image_fixture(
        tmp_path,
        indexed=True,
        architecture="arm64" if change == "architecture" else "amd64",
        ambiguous=change == "ambiguous",
    )
    if change in {"config_bytes", "manifest_bytes", "tag_bytes"}:
        name = {
            "config_bytes": "image-config.json",
            "manifest_bytes": "platform-manifest.json",
            "tag_bytes": "tag-manifest.json",
        }[change]
        with (tmp_path / name).open("ab") as f:
            f.write(b" ")
    elif change == "config_digest":
        receipt["config_digest"] = "sha256:" + "b" * 64
    elif change == "layers":
        receipt["layers"] = {}
    elif change == "total":
        receipt["compressed_layer_bytes"] = 0
    with pytest.raises(ValueError):
        audit.audit_image(tmp_path, receipt)


def task_fixture(tmp_path):
    import tomllib

    task = """schema_version="1.1"
[task]
name="terminal-bench/fixture"
[environment]
docker_image="alexgshaw/fixture:20260430"
cpus=2
memory_mb=4096
storage_mb=10240
gpus=0
build_timeout_sec=600.0
allow_internet=true
mcp_servers=[]
[environment.env]
PRIVATE_VALUE="synthetic-value-not-for-publication"
[agent]
timeout_sec=900.0
[verifier]
timeout_sec=600.0
"""
    folder = tmp_path / "snapshot/tasks/fixture"
    folder.mkdir(parents=True)
    (folder / "task.toml").write_text(task)
    files = {
        "tasks/fixture/" + name: {
            "sha256": audit.sha(task.encode())
            if name == "task.toml"
            else "fixture-hash"
        }
        for name in [
            "task.toml",
            "instruction.md",
            "environment/Dockerfile",
            "tests/test.sh",
            "solution/solve.sh",
        ]
    }
    receipt = image_fixture(tmp_path / "image-audit/fixture")
    parsed = tomllib.loads(task)
    receipt.update({key: parsed[key] for key in ["environment", "agent", "verifier"]})
    receipt.update(
        task_id="fixture",
        task_toml_sha256=audit.sha(task.encode()),
        image=parsed["environment"]["docker_image"],
    )
    path = tmp_path / "image-audit/fixture.json"
    path.write_text(json.dumps(receipt))
    return files, receipt, path


def test_task_uses_frozen_toml_tag_and_redacts_environment(tmp_path):
    files, _, _ = task_fixture(tmp_path)
    result = audit.audit_task("fixture", tmp_path, files)
    assert result["image"].endswith(":20260430")
    assert "synthetic-value-not-for-publication" not in json.dumps(result)
    assert result["task_executed"] is False


@pytest.mark.parametrize(
    "change", ["image", "identity", "resource", "toml_hash", "missing_file"]
)
def test_task_rejects_mapping_changes(tmp_path, change):
    files, receipt, path = task_fixture(tmp_path)
    if change == "image":
        receipt["image"] += "-changed"
    elif change == "identity":
        receipt["task_id"] = "different"
    elif change == "resource":
        receipt["environment"]["cpus"] = 4
    elif change == "toml_hash":
        receipt["task_toml_sha256"] = "b" * 64
    elif change == "missing_file":
        del files["tasks/fixture/tests/test.sh"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        audit.audit_task("fixture", tmp_path, files)
