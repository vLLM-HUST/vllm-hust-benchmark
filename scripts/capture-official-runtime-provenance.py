#!/usr/bin/env python3
"""Capture and validate the Python runtime used by an official benchmark."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

SCHEMA_VERSION = "official-runtime-provenance/v1"
ROLE_CONFIG = {
    "engine": {
        "module": "vllm",
        "distributions": ("vllm",),
        "extension_modules": ("vllm._C", "vllm._C_stable_libtorch"),
    },
    "plugin": {
        "module": "vllm_ascend",
        "distributions": ("vllm-ascend", "vllm-ascend-hust", "vllm_ascend"),
        "extension_modules": ("vllm_ascend.vllm_ascend_C",),
    },
}


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _canonical_version(value: str) -> str:
    normalized = value.strip().lower().removeprefix("v")
    match = re.fullmatch(r"(.+)-(\d+)-g([0-9a-f]+)(?:-dirty)?", normalized)
    if match:
        normalized = f"{match.group(1)}.dev{match.group(2)}+g{match.group(3)}"
    return normalized.replace("-", ".")


def _embedded_commit(version: str) -> str:
    match = re.search(r"(?:^|[.+-])g([0-9a-f]{7,40})(?:[.+-]|$)", version.lower())
    return match.group(1) if match else ""


def _distribution_version(names: tuple[str, ...]) -> tuple[str, str]:
    for name in names:
        try:
            return name, importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "", ""


def _module_path(module: ModuleType) -> Path:
    raw_path = getattr(module, "__file__", "")
    if not raw_path:
        raise ValueError(f"imported module {module.__name__} has no __file__")
    return Path(raw_path).resolve()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extension_evidence(
    namespace: str, candidates: tuple[str, ...], worktree: Path
) -> list[dict[str, Any]]:
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    paths: dict[str, tuple[str, Path]] = {}
    for name, module in tuple(sys.modules.items()):
        raw_path = getattr(module, "__file__", "") if module else ""
        if (
            name.startswith(f"{namespace}.")
            and raw_path
            and raw_path.endswith(suffixes)
        ):
            paths[name] = ("loaded", Path(raw_path).resolve())
    for name in candidates:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is not None and spec.origin and spec.origin.endswith(suffixes):
            paths.setdefault(name, ("resolved", Path(spec.origin).resolve()))
    package_root = worktree / namespace
    if package_root.is_dir():
        for path in package_root.rglob("*"):
            if path.is_file() and path.name.endswith(suffixes):
                relative = path.relative_to(worktree).as_posix()
                paths.setdefault(relative, ("discovered", path.resolve()))
    return [
        {
            "module": name,
            "status": status,
            "path": str(path),
            "sha256": _sha256(path),
        }
        for name, (status, path) in sorted(paths.items())
    ]


def capture_role(
    role: str,
    worktree: Path,
    expected_commit: str,
    *,
    runtime_root: Path | None = None,
    image_commit: str = "",
    image_id: str = "",
) -> dict[str, Any]:
    config = ROLE_CONFIG[role]
    worktree = worktree.resolve()
    observed_commit = _run_git(worktree, "rev-parse", "--verify", "HEAD^{commit}")
    expected_commit = expected_commit.strip().lower()
    if observed_commit != expected_commit:
        raise ValueError(
            f"{role} prepared worktree commit mismatch: expected {expected_commit}, "
            f"observed {observed_commit}"
        )

    module = importlib.import_module(config["module"])
    module_path = _module_path(module)
    runtime_root = (runtime_root or worktree).resolve()
    if not _is_relative_to(module_path, runtime_root):
        raise ValueError(
            f"{role} module path mismatch: {module_path} is outside runtime root "
            f"{runtime_root}"
        )

    image_commit = image_commit.strip().lower()
    image_id = image_id.strip()
    image_native = runtime_root != worktree
    if image_native:
        if not image_id:
            raise ValueError(
                f"{role} image-native runtime is missing immutable image id"
            )
        if image_commit != expected_commit:
            raise ValueError(
                f"{role} image commit mismatch: expected {expected_commit}, "
                f"observed {image_commit or '<empty>'}"
            )

    module_version = str(getattr(module, "__version__", "") or "").strip()
    module_commit = str(getattr(module, "__commit_id__", "") or "").strip()
    upstream_commit = str(getattr(module, "__upstream_commit__", "") or "").strip()
    distribution, distribution_version = _distribution_version(config["distributions"])
    if not module_version or not distribution_version:
        raise ValueError(f"{role} runtime package version cannot be proven")
    if _canonical_version(module_version) != _canonical_version(distribution_version):
        raise ValueError(
            f"{role} runtime version mismatch: module={module_version!r}, "
            f"distribution={distribution_version!r}"
        )

    source_version = _run_git(worktree, "describe", "--tags", "--always", "HEAD")
    source_commit_hint = _embedded_commit(source_version)
    runtime_commit_hint = _embedded_commit(module_version)
    module_commit_hint = module_commit.lower().removeprefix("g")
    if module_commit_hint:
        if not observed_commit.startswith(module_commit_hint):
            raise ValueError(
                f"{role} runtime build commit mismatch: module commit={module_commit!r}, "
                f"prepared commit={observed_commit}"
            )
    elif source_commit_hint and runtime_commit_hint:
        if not observed_commit.startswith(
            source_commit_hint
        ) or not observed_commit.startswith(runtime_commit_hint):
            raise ValueError(
                f"{role} runtime build commit mismatch: source={source_version!r}, "
                f"module={module_version!r}, commit={observed_commit}"
            )
    elif not image_native and _canonical_version(source_version) != _canonical_version(
        module_version
    ):
        raise ValueError(
            f"{role} runtime version does not identify prepared source: "
            f"source={source_version!r}, module={module_version!r}"
        )

    extensions = _extension_evidence(
        config["module"], config["extension_modules"], runtime_root
    )
    for extension in extensions:
        extension_path = Path(extension["path"])
        if not _is_relative_to(extension_path, runtime_root):
            raise ValueError(
                f"{role} extension path mismatch: {extension_path} is outside "
                f"runtime root {runtime_root}"
            )

    return {
        "module": config["module"],
        "module_path": str(module_path),
        "module_version": module_version,
        "module_commit": module_commit,
        "upstream_commit": upstream_commit,
        "distribution": distribution,
        "distribution_version": distribution_version,
        "prepared_worktree": str(worktree),
        "prepared_commit": observed_commit,
        "runtime_root": str(runtime_root),
        "runtime_binding": "image-native-oci" if image_native else "prepared-worktree",
        "runtime_image_id": image_id,
        "runtime_image_commit": image_commit,
        "source_version": source_version,
        "extension_policy": "present" if extensions else "none-discovered",
        "extensions": extensions,
    }


def _load_source_provenance(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "official-source-provenance/v1":
        raise ValueError(f"unsupported official source provenance: {path}")
    return payload


def capture(
    engine_worktree: Path,
    engine_commit: str,
    plugin_worktree: Path,
    plugin_commit: str,
    source_provenance: dict[str, Any] | None = None,
    engine_runtime_root: Path | None = None,
    plugin_runtime_root: Path | None = None,
    runtime_image_id: str = "",
    engine_image_commit: str = "",
    plugin_image_commit: str = "",
) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": sys.version.split()[0],
        "sources": {
            "engine": capture_role(
                "engine",
                engine_worktree,
                engine_commit,
                runtime_root=engine_runtime_root,
                image_commit=engine_image_commit,
                image_id=runtime_image_id,
            ),
            "plugin": capture_role(
                "plugin",
                plugin_worktree,
                plugin_commit,
                runtime_root=plugin_runtime_root,
                image_commit=plugin_image_commit,
                image_id=runtime_image_id,
            ),
        },
    }
    if source_provenance is not None:
        for role in ("engine", "plugin"):
            source = (source_provenance.get("sources") or {}).get(role) or {}
            runtime = payload["sources"][role]
            if source.get("observed_commit") != runtime["prepared_commit"]:
                raise ValueError(
                    f"{role} source/runtime commit mismatch: "
                    f"source={source.get('observed_commit')!r}, "
                    f"runtime={runtime['prepared_commit']!r}"
                )
            for source_field, runtime_field in (
                ("tracked_patch_sha256", "source_patch_sha256"),
                ("working_tree_sha256", "source_tree_sha256"),
                ("status", "source_status"),
            ):
                value = source.get(source_field)
                if not value:
                    raise ValueError(
                        f"{role} source provenance is missing {source_field}"
                    )
                runtime[runtime_field] = value
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-worktree", type=Path, required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--plugin-worktree", type=Path, required=True)
    parser.add_argument("--plugin-commit", required=True)
    parser.add_argument("--source-provenance", type=Path)
    parser.add_argument("--engine-runtime-root", type=Path)
    parser.add_argument("--plugin-runtime-root", type=Path)
    parser.add_argument("--runtime-image-id", default="")
    parser.add_argument("--engine-image-commit", default="")
    parser.add_argument("--plugin-image-commit", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        source_provenance = (
            _load_source_provenance(args.source_provenance)
            if args.source_provenance
            else None
        )
        payload = capture(
            args.engine_worktree,
            args.engine_commit,
            args.plugin_worktree,
            args.plugin_commit,
            source_provenance,
            args.engine_runtime_root,
            args.plugin_runtime_root,
            args.runtime_image_id,
            args.engine_image_commit,
            args.plugin_image_commit,
        )
    except (ImportError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(
            f"official runtime provenance validation failed: {error}", file=sys.stderr
        )
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
