#!/usr/bin/env python3
"""Create deterministic Issue 214 reference runtime and cell contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


CORE_COMMIT = "43341b177dbaa8c7f04662f71e885ee7dfe22704"  # pragma: allowlist secret
PLUGIN_COMMIT = "0a46364814eedd3314f04eff3490c3ab422438bd"  # pragma: allowlist secret
IMAGE_ID = "sha256:cd3c36c1832d8b9dfd193405d52a73b22ca2699fbe95271015bc7932bef3e2ee"  # pragma: allowlist secret


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git(repo: Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=text)


def require_commit(repo: Path, expected: str) -> None:
    observed = str(git(repo, "rev-parse", "HEAD")).strip()
    if observed != expected:
        raise ValueError(f"{repo}: expected {expected}, observed {observed}")


def file_manifest(root: Path, patterns: tuple[str, ...]) -> list[dict[str, Any]]:
    paths = {
        path.resolve()
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
    }
    return [
        {
            "path": path.relative_to(root.resolve()).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]


def package_version(python: Path, name: str) -> str:
    code = f"from importlib.metadata import version; print(version({name!r}))"
    return subprocess.check_output([str(python), "-c", code], text=True).strip()


def runtime_lock(args: argparse.Namespace) -> None:
    require_commit(args.source_core, CORE_COMMIT)
    require_commit(args.source_plugin, PLUGIN_COMMIT)
    require_commit(args.runtime_core, CORE_COMMIT)
    require_commit(args.runtime_plugin, PLUGIN_COMMIT)
    for repo in (args.source_core, args.source_plugin):
        if str(git(repo, "status", "--porcelain")).strip():
            raise ValueError(f"frozen source worktree is dirty: {repo}")

    core_patch = bytes(git(args.runtime_core, "diff", "--binary", text=False))
    plugin_patch = bytes(git(args.runtime_plugin, "diff", "--binary", text=False))
    core_artifacts = file_manifest(
        args.runtime_core,
        ("vllm/_version.py", "vllm-*.dist-info/*"),
    )
    artifacts = file_manifest(
        args.runtime_plugin,
        (
            "vllm_ascend/**/*.so",
            "vllm_ascend/**/*.o",
            "vllm_ascend/_official_wheel_artifacts.json",
            "vllm_ascend/_build_info.py",
            "vllm_ascend/_version.py",
            "vllm_ascend-*.dist-info/*",
        ),
    )
    if not artifacts or not any(item["path"].endswith(".so") for item in artifacts):
        raise ValueError("plugin runtime has no frozen compiled extension artifacts")
    artifact_digest = hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "schema_version": "issue214-reference-runtime-lock/v1",
        "status": "ok",
        "image_id": IMAGE_ID,
        "cann": args.cann,
        "python": subprocess.check_output(
            [
                str(args.python),
                "-c",
                "import platform; print(platform.python_version())",
            ],
            text=True,
        ).strip(),
        "torch": package_version(args.python, "torch"),
        "torch_npu": package_version(args.python, "torch-npu"),
        "sources": {
            "core": {
                "commit": CORE_COMMIT,
                "tree": str(git(args.source_core, "rev-parse", "HEAD^{tree}")).strip(),
                "remote": args.core_remote,
            },
            "plugin": {
                "commit": PLUGIN_COMMIT,
                "tree": str(
                    git(args.source_plugin, "rev-parse", "HEAD^{tree}")
                ).strip(),
                "remote": args.plugin_remote,
            },
        },
        "compatibility_overlay": {
            "core_tracked_patch_sha256": hashlib.sha256(core_patch).hexdigest(),
            "core_tracked_patch_bytes": len(core_patch),
            "plugin_tracked_patch_sha256": hashlib.sha256(plugin_patch).hexdigest(),
            "plugin_tracked_patch_bytes": len(plugin_patch),
            "core_artifacts": core_artifacts,
            "artifact_manifest_sha256": artifact_digest,
            "artifacts": artifacts,
        },
    }
    write_json(args.output, payload)


def model_manifest(args: argparse.Namespace) -> None:
    if not args.model.is_dir():
        raise ValueError(f"model directory is missing: {args.model}")
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    files = [
        {
            "path": name,
            "sha256": sha256(args.model / name),
            "size_bytes": (args.model / name).stat().st_size,
        }
        for name in names
        if (args.model / name).is_file()
    ]
    if not any(item["path"] == "config.json" for item in files):
        raise ValueError(f"model config is missing: {args.model / 'config.json'}")
    weight_files = sorted(args.model.glob("*.safetensors"))
    if not weight_files:
        raise ValueError(f"model weights are missing: {args.model}")
    weights = [
        {
            "path": path.name,
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in weight_files
    ]
    write_json(
        args.output,
        {
            "schema_version": "issue214-model-manifest/v1",
            "canonical_id": args.canonical_id,
            "files": files,
            "weights": weights,
        },
    )


def normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: normalized(item)
            for key, item in sorted(value.items())
            if key not in {"host", "port", "model", "dataset_path", "spec_source"}
        }
    if isinstance(value, list):
        return [normalized(item) for item in value]
    return value


def archive_cell(args: argparse.Namespace) -> None:
    runtime_lock_payload = json.loads(args.runtime_lock.read_text())
    if runtime_lock_payload.get("status") != "ok":
        raise ValueError("runtime lock is not qualified")
    resolved = json.loads((args.cell / "resolved_same_spec.json").read_text())
    run = json.loads((args.cell / "submission/run_leaderboard.json").read_text())
    if normalized(resolved) != normalized(run.get("same_spec", {})):
        raise ValueError("resolved_same_spec does not match exported run")

    runtime_contract = {
        key: runtime_lock_payload[key]
        for key in ("image_id", "cann", "python", "torch", "torch_npu")
    }
    runtime_contract["schema_version"] = "issue214-runtime-contract/v1"
    write_json(args.cell / "runtime-contract.json", runtime_contract)

    provenance_path = args.cell / "submission/input_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    )
    input_identity = {
        "schema_version": "issue214-input-identity/v1",
        "spec_sha256": sha256(args.spec),
        "resolved_workload": normalized(resolved),
        "input_provenance": normalized(provenance),
        "model_manifest": json.loads(args.model_manifest.read_text()),
    }
    write_json(args.cell / "input-identity.json", input_identity)

    required = {
        "raw_benchmark_result.json",
        "resolved_same_spec.json",
        "runner.log",
        "runtime-contract.json",
        "input-identity.json",
        "submission/STATUS",
        "submission/checksums.sha256",
        "submission/env-manifest.json",
        "submission/leaderboard_manifest.json",
        "submission/pip-packages.json",
        "submission/run_leaderboard.json",
    }
    if run.get("execution_mode") == "online" or resolved.get("scenario", "").endswith(
        "online"
    ):
        required.add("server.stdout.log")
    missing = sorted(name for name in required if not (args.cell / name).is_file())
    if missing:
        raise ValueError(f"cell archive is incomplete: {missing}")
    lines = [f"{sha256(args.cell / name)}  {name}" for name in sorted(required)]
    (args.cell / "EVIDENCE_SHA256SUMS").write_text("\n".join(lines) + "\n")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(required=True)
    lock = commands.add_parser("runtime-lock")
    lock.add_argument("--source-core", required=True, type=Path)
    lock.add_argument("--source-plugin", required=True, type=Path)
    lock.add_argument("--runtime-core", required=True, type=Path)
    lock.add_argument("--runtime-plugin", required=True, type=Path)
    lock.add_argument("--python", required=True, type=Path)
    lock.add_argument("--cann", required=True)
    lock.add_argument("--core-remote", required=True)
    lock.add_argument("--plugin-remote", required=True)
    lock.add_argument("--output", required=True, type=Path)
    lock.set_defaults(handler=runtime_lock)

    model = commands.add_parser("model-manifest")
    model.add_argument("--model", required=True, type=Path)
    model.add_argument("--canonical-id", required=True)
    model.add_argument("--output", required=True, type=Path)
    model.set_defaults(handler=model_manifest)

    cell = commands.add_parser("archive-cell")
    cell.add_argument("--cell", required=True, type=Path)
    cell.add_argument("--spec", required=True, type=Path)
    cell.add_argument("--runtime-lock", required=True, type=Path)
    cell.add_argument("--model-manifest", required=True, type=Path)
    cell.set_defaults(handler=archive_cell)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
