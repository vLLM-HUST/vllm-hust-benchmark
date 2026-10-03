#!/usr/bin/env python3
"""Restore official vLLM Ascend binary artifacts into a source worktree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wheel = args.wheel.resolve()
    worktree = args.worktree.resolve()
    expected_digest = args.expected_sha256.lower()
    observed_digest = sha256(wheel)
    if observed_digest != expected_digest:
        raise SystemExit(
            f"wheel SHA256 mismatch: observed={observed_digest} expected={expected_digest}"
        )

    package_prefix = PurePosixPath("vllm_ascend")
    custom_prefix = package_prefix / "_cann_ops_custom"
    selected: list[tuple[object, PurePosixPath]] = []
    metadata_text: str | None = None

    with ZipFile(wheel) as archive:
        for info in archive.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise SystemExit(f"unsafe wheel member: {info.filename}")
            if member.name == "METADATA" and member.parent.name.endswith(".dist-info"):
                metadata_text = archive.read(info).decode("utf-8")
            is_extension = (
                member.parent == package_prefix
                and (
                    member.name.startswith("vllm_ascend_C")
                    or member.name == "libvllm_ascend_kernels.so"
                )
                and member.suffix == ".so"
            )
            is_custom_op = member == custom_prefix or custom_prefix in member.parents
            if is_extension or is_custom_op:
                selected.append((info, member))

        version_line = next(
            (
                line
                for line in (metadata_text or "").splitlines()
                if line.startswith("Version: ")
            ),
            "",
        )
        observed_version = version_line.partition(": ")[2]
        if observed_version != args.expected_version:
            raise SystemExit(
                "wheel version mismatch: "
                f"observed={observed_version or '<missing>'} "
                f"expected={args.expected_version}"
            )

        if not any(member.name.startswith("vllm_ascend_C") for _, member in selected):
            raise SystemExit("wheel contains no vllm_ascend_C extension")
        if not any(custom_prefix in member.parents for _, member in selected):
            raise SystemExit("wheel contains no packaged custom operators")

        extracted: list[dict[str, object]] = []
        for info, member in selected:
            target = worktree.joinpath(*member.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.append(
                {
                    "path": member.as_posix(),
                    "size_bytes": target.stat().st_size,
                    "sha256": sha256(target),
                }
            )

    manifest = {
        "schema_version": "official-vllm-ascend-wheel-extraction/v1",
        "wheel": str(wheel),
        "wheel_version": observed_version,
        "wheel_sha256": observed_digest,
        "artifact_count": len(extracted),
        "artifacts": sorted(extracted, key=lambda item: str(item["path"])),
    }
    manifest_path = worktree / "vllm_ascend" / "_official_wheel_artifacts.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
