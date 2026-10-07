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
    resolved_root = root.resolve()
    paths = {
        path for pattern in patterns for path in root.glob(pattern) if path.is_file()
    }
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(resolved_root):
            raise ValueError(f"unsafe runtime artifact path: {path}")
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(paths)
    ]


def untracked_manifest(repo: Path) -> list[dict[str, Any]]:
    output = bytes(
        git(repo, "ls-files", "--others", "--exclude-standard", "-z", text=False)
    )
    paths = [repo / item.decode() for item in output.split(b"\0") if item]
    return file_manifest(
        repo, tuple(path.relative_to(repo).as_posix() for path in paths)
    )


def package_version(python: Path, name: str) -> str:
    code = f"from importlib.metadata import version; print(version({name!r}))"
    return subprocess.check_output([str(python), "-c", code], text=True).strip()


def build_runtime_lock(args: argparse.Namespace) -> dict[str, Any]:
    require_commit(args.source_core, args.core_commit)
    require_commit(args.source_plugin, args.plugin_commit)
    require_commit(args.runtime_core, args.core_commit)
    require_commit(args.runtime_plugin, args.plugin_commit)
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
    custom_op_artifacts = file_manifest(
        args.runtime_plugin,
        ("vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/**/*",),
    )
    if not artifacts or not any(item["path"].endswith(".so") for item in artifacts):
        raise ValueError("plugin runtime has no frozen compiled extension artifacts")
    if not custom_op_artifacts:
        raise ValueError("plugin runtime has no frozen custom-op artifacts")
    core_untracked = untracked_manifest(args.runtime_core)
    plugin_untracked = untracked_manifest(args.runtime_plugin)
    artifact_digest = hashlib.sha256(
        json.dumps(
            {
                "core_artifacts": core_artifacts,
                "core_untracked_files": core_untracked,
                "plugin_artifacts": artifacts,
                "plugin_untracked_files": plugin_untracked,
                "custom_op_artifacts": custom_op_artifacts,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema_version": "issue214-runtime-lock/v2",
        "status": "ok",
        "role": args.role,
        "image_id": args.image_id,
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
                "commit": args.core_commit,
                "ref": args.core_ref,
                "tree": str(git(args.source_core, "rev-parse", "HEAD^{tree}")).strip(),
                "remote": args.core_remote,
            },
            "plugin": {
                "commit": args.plugin_commit,
                "ref": args.plugin_ref,
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
            "core_untracked_files": core_untracked,
            "plugin_untracked_files": plugin_untracked,
            "artifact_manifest_sha256": artifact_digest,
            "artifacts": artifacts,
            "custom_op_artifacts": custom_op_artifacts,
        },
    }


def runtime_lock(args: argparse.Namespace) -> None:
    write_json(args.output, build_runtime_lock(args))


def verify_runtime_lock(args: argparse.Namespace) -> None:
    expected = json.loads(args.lock.read_text())
    observed = build_runtime_lock(args)
    if observed != expected:
        raise ValueError("runtime sources, overlay, packages, or artifacts changed")


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
    if (
        runtime_lock_payload.get("schema_version") != "issue214-runtime-lock/v2"
        or runtime_lock_payload.get("status") != "ok"
    ):
        raise ValueError("runtime lock is not qualified")
    if runtime_lock_payload.get("role") != args.role:
        raise ValueError("runtime lock role does not match archive role")
    if not args.spec.is_file() or args.spec.is_symlink():
        raise ValueError("spec is missing or unsafe")
    spec_payload = json.loads(args.spec.read_text())
    model_payload = json.loads(args.model_manifest.read_text())
    if model_payload.get(
        "schema_version"
    ) != "issue214-model-manifest/v1" or model_payload.get(
        "canonical_id"
    ) != spec_payload.get("model"):
        raise ValueError("model manifest does not match spec model identity")
    resolved = json.loads((args.cell / "resolved_same_spec.json").read_text())
    run = json.loads((args.cell / "submission/run_leaderboard.json").read_text())
    if normalized(resolved) != normalized(run.get("same_spec", {})):
        raise ValueError("resolved_same_spec does not match exported run")

    runtime_contract = {
        key: runtime_lock_payload[key]
        for key in ("image_id", "cann", "python", "torch", "torch_npu")
    }
    runtime_contract["schema_version"] = "issue214-runtime-contract/v2"
    runtime_contract["role"] = runtime_lock_payload.get("role")
    runtime_contract["runtime_lock_sha256"] = sha256(args.runtime_lock)
    runtime_contract["sources"] = runtime_lock_payload.get("sources")
    runtime_contract["compatibility_overlay"] = {
        key: runtime_lock_payload["compatibility_overlay"].get(key)
        for key in (
            "core_tracked_patch_sha256",
            "plugin_tracked_patch_sha256",
            "artifact_manifest_sha256",
        )
    }
    write_json(args.cell / "runtime-contract.json", runtime_contract)

    provenance_path = args.cell / "submission/input_provenance.json"
    provenance = (
        json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
    )
    input_files: list[dict[str, Any]] = []
    input_labels: set[str] = set()
    for declaration in args.input_file:
        label, separator, raw_path = declaration.partition("=")
        path = Path(raw_path)
        if not separator or not label or not path.is_file() or path.is_symlink():
            raise ValueError(f"invalid frozen input declaration: {declaration}")
        if label in input_labels:
            raise ValueError(f"duplicate frozen input label: {label}")
        input_labels.add(label)
        input_files.append(
            {"label": label, "sha256": sha256(path), "size_bytes": path.stat().st_size}
        )
    input_identity = {
        "schema_version": "issue214-input-identity/v1",
        "spec_sha256": sha256(args.spec),
        "resolved_workload": normalized(resolved),
        "input_provenance": normalized(provenance),
        "input_files": sorted(input_files, key=lambda item: item["label"]),
        "synthetic_generation": {
            "seed": resolved.get("resolved_client_parameters", {}).get("seed", 0),
            "parameters": normalized(resolved.get("resolved_client_parameters", {})),
        },
        "model_manifest": model_payload,
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
    archived_files = {
        path.relative_to(args.cell).as_posix()
        for path in args.cell.rglob("*")
        if path.is_file() and path.name != "EVIDENCE_SHA256SUMS"
    }
    unsafe = [
        name
        for name in archived_files
        if (args.cell / name).is_symlink()
        or not (args.cell / name).resolve().is_relative_to(args.cell.resolve())
    ]
    if unsafe:
        raise ValueError(f"cell archive contains unsafe files: {sorted(unsafe)}")
    lines = [f"{sha256(args.cell / name)}  {name}" for name in sorted(archived_files)]
    evidence_manifest = args.cell / "EVIDENCE_SHA256SUMS"
    temporary = evidence_manifest.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(evidence_manifest)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(required=True)

    def add_runtime_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--source-core", required=True, type=Path)
        command.add_argument("--source-plugin", required=True, type=Path)
        command.add_argument("--runtime-core", required=True, type=Path)
        command.add_argument("--runtime-plugin", required=True, type=Path)
        command.add_argument("--python", required=True, type=Path)
        command.add_argument("--cann", required=True)
        command.add_argument("--core-remote", required=True)
        command.add_argument("--plugin-remote", required=True)
        command.add_argument(
            "--role", choices=("reference", "candidate"), default="reference"
        )
        command.add_argument("--core-commit", default=CORE_COMMIT)
        command.add_argument("--plugin-commit", default=PLUGIN_COMMIT)
        command.add_argument("--core-ref", default=CORE_COMMIT)
        command.add_argument("--plugin-ref", default=PLUGIN_COMMIT)
        command.add_argument("--image-id", default=IMAGE_ID)

    lock = commands.add_parser("runtime-lock")
    add_runtime_arguments(lock)
    lock.add_argument("--output", required=True, type=Path)
    lock.set_defaults(handler=runtime_lock)

    verify = commands.add_parser("verify-runtime-lock")
    add_runtime_arguments(verify)
    verify.add_argument("--lock", required=True, type=Path)
    verify.set_defaults(handler=verify_runtime_lock)

    model = commands.add_parser("model-manifest")
    model.add_argument("--model", required=True, type=Path)
    model.add_argument("--canonical-id", required=True)
    model.add_argument("--output", required=True, type=Path)
    model.set_defaults(handler=model_manifest)

    cell = commands.add_parser("archive-cell")
    cell.add_argument("--role", required=True, choices=("reference", "candidate"))
    cell.add_argument("--cell", required=True, type=Path)
    cell.add_argument("--spec", required=True, type=Path)
    cell.add_argument("--runtime-lock", required=True, type=Path)
    cell.add_argument("--model-manifest", required=True, type=Path)
    cell.add_argument("--input-file", action="append", default=[])
    cell.set_defaults(handler=archive_cell)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
