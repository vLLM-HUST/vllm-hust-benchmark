from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/issue214_reference_archive.py"


@pytest.fixture
def helper():
    spec = importlib.util.spec_from_file_location("issue214_reference_archive", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def git_repo(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.com"],
        check=True,
    )
    (path / "tracked").write_text("base\n")
    subprocess.run(["git", "-C", str(path), "add", "tracked"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "base"], check=True)
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def test_runtime_lock_binds_clean_sources_overlay_and_binary(tmp_path, helper):
    source_core = tmp_path / "source-core"
    source_plugin = tmp_path / "source-plugin"
    runtime_core = tmp_path / "runtime-core"
    runtime_plugin = tmp_path / "runtime-plugin"
    core_commit = git_repo(source_core)
    plugin_commit = git_repo(source_plugin)
    subprocess.run(
        ["git", "clone", "-q", str(source_core), str(runtime_core)], check=True
    )
    subprocess.run(
        ["git", "clone", "-q", str(source_plugin), str(runtime_plugin)], check=True
    )
    (runtime_core / "tracked").write_text("compat\n")
    (runtime_plugin / "tracked").write_text("compat\n")
    extension = runtime_plugin / "vllm_ascend/runtime.so"
    extension.parent.mkdir()
    extension.write_bytes(b"binary")
    custom_op = (
        runtime_plugin
        / "vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_impl/custom.py"
    )
    custom_op.parent.mkdir(parents=True)
    custom_op.write_text("# frozen custom op\n")
    helper.CORE_COMMIT = core_commit
    helper.PLUGIN_COMMIT = plugin_commit
    output = tmp_path / "lock.json"

    helper.runtime_lock(
        Namespace(
            source_core=source_core,
            source_plugin=source_plugin,
            runtime_core=runtime_core,
            runtime_plugin=runtime_plugin,
            python=Path(sys.executable),
            cann="9.1",
            core_remote="core",
            plugin_remote="plugin",
            role="reference",
            core_commit=core_commit,
            plugin_commit=plugin_commit,
            core_ref=core_commit,
            plugin_ref=plugin_commit,
            image_id="sha256:image",
            output=output,
        )
    )

    payload = json.loads(output.read_text())
    assert payload["status"] == "ok"
    assert payload["compatibility_overlay"]["core_tracked_patch_bytes"] > 0
    assert payload["compatibility_overlay"]["plugin_tracked_patch_bytes"] > 0
    assert payload["compatibility_overlay"]["artifacts"][0]["path"].endswith(
        "runtime.so"
    )
    assert payload["compatibility_overlay"]["custom_op_artifacts"]

    helper.verify_runtime_lock(
        Namespace(
            source_core=source_core,
            source_plugin=source_plugin,
            runtime_core=runtime_core,
            runtime_plugin=runtime_plugin,
            python=Path(sys.executable),
            cann="9.1",
            core_remote="core",
            plugin_remote="plugin",
            role="reference",
            core_commit=core_commit,
            plugin_commit=plugin_commit,
            core_ref=core_commit,
            plugin_ref=plugin_commit,
            image_id="sha256:image",
            lock=output,
        )
    )
    extension.write_bytes(b"changed")
    with pytest.raises(ValueError, match="runtime sources"):
        helper.verify_runtime_lock(
            Namespace(
                source_core=source_core,
                source_plugin=source_plugin,
                runtime_core=runtime_core,
                runtime_plugin=runtime_plugin,
                python=Path(sys.executable),
                cann="9.1",
                core_remote="core",
                plugin_remote="plugin",
                role="reference",
                core_commit=core_commit,
                plugin_commit=plugin_commit,
                core_ref=core_commit,
                plugin_ref=plugin_commit,
                image_id="sha256:image",
                lock=output,
            )
        )


def test_archive_cell_writes_identity_contract_and_complete_checksums(tmp_path, helper):
    cell = tmp_path / "cell"
    submission = cell / "submission"
    submission.mkdir(parents=True)
    resolved = {"scenario": "random-online", "port": 8000, "model": "/models/x"}
    write_json(cell / "resolved_same_spec.json", resolved)
    write_json(cell / "raw_benchmark_result.json", {"ok": True})
    write_json(cell / "runtime-ready.log", {})
    (cell / "runner.log").write_text("ok\n")
    (cell / "server.stdout.log").write_text("ready\n")
    for name in (
        "env-manifest.json",
        "leaderboard_manifest.json",
        "pip-packages.json",
    ):
        write_json(submission / name, {})
    write_json(
        submission / "run_leaderboard.json",
        {"execution_mode": "online", "same_spec": resolved},
    )
    (submission / "STATUS").write_text("OK\n")
    (submission / "checksums.sha256").write_text("placeholder\n")
    runtime_lock = tmp_path / "runtime-lock.json"
    write_json(
        runtime_lock,
        {
            "schema_version": "issue214-runtime-lock/v2",
            "status": "ok",
            "image_id": "sha256:image",
            "cann": "9.1",
            "python": "3.11",
            "torch": "2.9.0",
            "torch_npu": "2.9.0",
            "sources": {"core": {"commit": "a" * 40}, "plugin": {"commit": "b" * 40}},
            "compatibility_overlay": {
                "core_tracked_patch_sha256": "1" * 64,
                "plugin_tracked_patch_sha256": "2" * 64,
                "artifact_manifest_sha256": "3" * 64,
            },
        },
    )
    spec = tmp_path / "spec.json"
    write_json(spec, {"scenario": "random-online"})
    model_manifest = tmp_path / "model.json"
    write_json(model_manifest, {"schema_version": "issue214-model-manifest/v1"})

    helper.archive_cell(
        Namespace(
            cell=cell,
            spec=spec,
            runtime_lock=runtime_lock,
            model_manifest=model_manifest,
            input_file=[],
        )
    )

    assert json.loads((cell / "runtime-contract.json").read_text())["cann"] == "9.1"
    assert json.loads((cell / "input-identity.json").read_text())["spec_sha256"]
    checksum_text = (cell / "EVIDENCE_SHA256SUMS").read_text()
    assert "server.stdout.log" in checksum_text
    assert "runtime-ready.log" in checksum_text
    subprocess.run(["sha256sum", "-c", "EVIDENCE_SHA256SUMS"], cwd=cell, check=True)
