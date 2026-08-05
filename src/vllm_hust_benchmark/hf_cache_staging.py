"""Stage verified flat Hugging Face exports as exact offline snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "hf-flat-cache-staging/v1"
REVISION_RE = re.compile(r"[0-9a-f]{40}")
REPOSITORY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
PARTIAL_SUFFIXES = (".incomplete", ".partial", ".lock", ".tmp")


class StagingFailure(RuntimeError):
    """The flat export cannot prove the requested immutable identity."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise StagingFailure(f"path escapes flat export: {path}") from error
    if relative.is_absolute() or ".." in relative.parts:
        raise StagingFailure(f"unsafe flat export path: {relative}")
    return relative


def _validate_identity(repository: str, revision: str) -> None:
    if not REPOSITORY_RE.fullmatch(repository):
        raise StagingFailure(
            f"invalid Hugging Face repository identity: {repository!r}"
        )
    if not REVISION_RE.fullmatch(revision):
        raise StagingFailure(f"revision is not an exact commit: {revision!r}")


def _metadata_revision(source: Path, relative: Path) -> str:
    metadata = (
        source
        / ".cache"
        / "huggingface"
        / "download"
        / Path(str(relative) + ".metadata")
    )
    if not metadata.is_file() or metadata.is_symlink():
        raise StagingFailure(f"original HF metadata is missing: {metadata}")
    try:
        first_line = metadata.read_text(encoding="utf-8").splitlines()[0]
    except (IndexError, OSError, UnicodeDecodeError) as error:
        raise StagingFailure(
            f"original HF metadata is malformed: {metadata}"
        ) from error
    return first_line


def _load_checksum_manifest(source: Path) -> dict[str, str] | None:
    path = source / "SHA256SUMS"
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise StagingFailure("SHA256SUMS must be a regular file")
    parsed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n\r]+)", line)
        if not match:
            raise StagingFailure("SHA256SUMS contains a malformed record")
        digest, name = match.groups()
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in parsed:
            raise StagingFailure(f"SHA256SUMS contains an unsafe path: {name}")
        parsed[name] = digest
    return parsed


def _validate_model_files(source: Path, files: set[str]) -> None:
    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index["weight_map"].values())
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise StagingFailure("model weight index is malformed") from error
        if not shards or any(
            not isinstance(shard, str)
            or Path(shard).is_absolute()
            or ".." in Path(shard).parts
            for shard in shards
        ):
            raise StagingFailure("model weight index contains unsafe shard paths")
        missing = sorted(shards - files)
        if missing:
            raise StagingFailure(
                f"model weight index references missing files: {missing}"
            )
    elif "model.safetensors" not in files:
        raise StagingFailure("model export has neither an index nor model.safetensors")
    if "config.json" not in files:
        raise StagingFailure("model export is missing config.json")


def inspect_flat_export(
    source_path: Path,
    *,
    repository: str,
    revision: str,
    repo_type: str,
) -> dict[str, Any]:
    """Hash and authenticate every exported HF file against original metadata."""
    _validate_identity(repository, revision)
    source = source_path.resolve(strict=True)
    if not source.is_dir():
        raise StagingFailure(f"flat export is not a directory: {source}")
    for path in source.rglob("*"):
        if any(path.name.endswith(suffix) for suffix in PARTIAL_SUFFIXES):
            raise StagingFailure(f"partial/lock artifact is present: {path}")
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            _safe_relative(resolved, source)
            raise StagingFailure(f"flat export symlinks are not accepted: {path}")

    files: list[dict[str, Any]] = []
    actual_names: set[str] = set()
    for path in sorted(source.rglob("*")):
        relative = _safe_relative(path, source)
        if relative.parts[0] == ".cache" or relative == Path("SHA256SUMS"):
            continue
        if not path.is_file():
            continue
        if _metadata_revision(source, relative) != revision:
            raise StagingFailure(f"HF metadata revision mismatch for {relative}")
        name = relative.as_posix()
        actual_names.add(name)
        files.append(
            {"path": name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    if not files:
        raise StagingFailure("flat export contains no authenticated files")
    metadata_root = source / ".cache" / "huggingface" / "download"
    metadata_names = {
        path.relative_to(metadata_root).as_posix()[: -len(".metadata")]
        for path in metadata_root.rglob("*.metadata")
        if path.is_file() and not path.is_symlink()
    }
    if metadata_names != actual_names:
        raise StagingFailure("original HF metadata file set does not match the export")

    checksum_manifest = _load_checksum_manifest(source)
    if checksum_manifest is not None:
        if set(checksum_manifest) != actual_names:
            raise StagingFailure("SHA256SUMS file set does not match the HF export")
        for record in files:
            if checksum_manifest[record["path"]] != record["sha256"]:
                raise StagingFailure(f"SHA256SUMS mismatch for {record['path']}")
    if repo_type == "model":
        _validate_model_files(source, actual_names)
    elif repo_type != "dataset":
        raise StagingFailure(f"unsupported repo type: {repo_type}")

    manifest_sha256 = hashlib.sha256(
        json.dumps(files, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return {
        "repository": repository,
        "revision": revision,
        "repo_type": repo_type,
        "source_path": str(source),
        "files": files,
        "file_count": len(files),
        "manifest_sha256": manifest_sha256,
        "source_sha256sums_verified": checksum_manifest is not None,
    }


def stage_flat_export(export: Mapping[str, Any], hub_cache: Path) -> dict[str, Any]:
    repository = str(export["repository"])
    revision = str(export["revision"])
    repo_type = str(export["repo_type"])
    source = Path(str(export["source_path"]))
    prefix = "models" if repo_type == "model" else "datasets"
    repository_cache = hub_cache / f"{prefix}--{repository.replace('/', '--')}"
    snapshot = repository_cache / "snapshots" / revision
    if snapshot.exists() or snapshot.is_symlink():
        raise StagingFailure(f"owned snapshot destination already exists: {snapshot}")
    snapshot.mkdir(parents=True)
    try:
        ref = repository_cache / "refs" / "main"
        ref.parent.mkdir(parents=True)
        ref.write_text(revision, encoding="utf-8")
        for record in export["files"]:
            relative = Path(record["path"])
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source / relative)
            metadata_relative = Path(".cache/huggingface/download") / relative
            metadata_destination = metadata_relative.with_name(
                metadata_relative.name + ".metadata"
            )
            staged_metadata = snapshot / metadata_destination
            staged_metadata.parent.mkdir(parents=True, exist_ok=True)
            staged_metadata.symlink_to(source / metadata_destination)
    except Exception:
        # The caller supplies a fresh owned scratch root; leave partial state visible.
        raise
    return {**dict(export), "snapshot_path": str(snapshot.resolve())}


def stage_huggingface_file(
    source_path: Path, dataset_root: Path, identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Bind one authenticated HF file into the owned runtime dataset cache."""
    if identity.get("kind") != "huggingface-file":
        raise StagingFailure("local HF file staging requires huggingface-file identity")
    repository = str(identity.get("repository") or "")
    revision = str(identity.get("revision") or "")
    _validate_identity(repository, revision)
    relative = Path(str(identity.get("path") or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise StagingFailure("huggingface-file identity contains an unsafe path")
    expected_size = identity.get("size_bytes")
    expected_sha256 = identity.get("sha256")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise StagingFailure("huggingface-file identity lacks an exact size")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise StagingFailure("huggingface-file identity lacks an exact SHA256")

    if source_path.is_symlink():
        raise StagingFailure("huggingface-file source must not be a symlink")
    source = source_path.resolve(strict=True)
    if not source.is_file():
        raise StagingFailure(f"huggingface-file source is not a file: {source}")
    actual_size = source.stat().st_size
    actual_sha256 = _sha256(source)
    if actual_size != expected_size or actual_sha256 != expected_sha256:
        raise StagingFailure("huggingface-file source identity mismatch")

    root = dataset_root.resolve()
    destination = root / relative
    if destination.exists() or destination.is_symlink():
        raise StagingFailure(
            f"owned huggingface-file destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        shutil.copyfile(source, temporary_name)
        staged = Path(temporary_name)
        if staged.stat().st_size != expected_size or _sha256(staged) != expected_sha256:
            raise StagingFailure("staged huggingface-file identity mismatch")
        with staged.open("r+b") as stream:
            os.fsync(stream.fileno())
        os.link(staged, destination)
        staged.unlink()
        temporary_name = None
        destination.chmod(0o444)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return {
        "kind": "huggingface-file",
        "repository": repository,
        "revision": revision,
        "path": relative.as_posix(),
        "size_bytes": actual_size,
        "sha256": actual_sha256,
        "source_path": str(source),
        "runtime_path": str(destination.resolve()),
    }


def _registry_target(repo_root: Path, spec_path: Path) -> dict[str, Any]:
    registry = json.loads(
        (repo_root / "src/vllm_hust_benchmark/data/official_targets.json").read_text(
            encoding="utf-8"
        )
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    relative_spec = spec_path.resolve().relative_to(repo_root.resolve()).as_posix()
    spec_sha256 = _sha256(spec_path)
    matches = [
        target
        for target in registry["targets"]
        if target.get("source_spec", {}).get("path") == relative_spec
        and target.get("source_spec", {}).get("sha256") == spec_sha256
    ]
    if len(matches) != 1:
        raise StagingFailure(
            "spec does not resolve to exactly one hashed registry target"
        )
    target = matches[0]
    if (
        spec.get("model") != target["model"].get("id")
        or spec.get("model_revision") != target["model"].get("revision")
        or spec.get("data_identity") != target["workload"].get("data_identity")
    ):
        raise StagingFailure(
            "spec model/data identity differs from the registry target"
        )
    return target


def target_data_identity_kind(repo_root: Path, spec_path: Path) -> str:
    """Return the data kind only after the spec matches its hashed registry target."""
    target = _registry_target(repo_root, spec_path)
    data = target["workload"].get("data_identity")
    if not isinstance(data, dict) or not isinstance(data.get("kind"), str):
        raise StagingFailure("official target lacks a data identity kind")
    return data["kind"]


def stage_target(
    *,
    repo_root: Path,
    spec_path: Path,
    scratch_root: Path,
    model_source: Path,
    dataset_source: Path | None,
    file_dataset_root: Path | None = None,
) -> dict[str, Any]:
    target = _registry_target(repo_root, spec_path)
    model = target["model"]
    data = target["workload"].get("data_identity")
    data_kind = data.get("kind") if isinstance(data, dict) else None
    if data_kind not in {"huggingface-dataset", "huggingface-file"}:
        if dataset_source is not None:
            raise StagingFailure("dataset source supplied for an unsupported target")
    elif dataset_source is None:
        raise StagingFailure(f"{data_kind} target requires a local dataset source")
    if data_kind == "huggingface-file" and file_dataset_root is None:
        raise StagingFailure("huggingface-file target requires a runtime dataset root")
    if scratch_root.exists():
        raise StagingFailure(f"staging scratch already exists: {scratch_root}")
    hub_cache = scratch_root / "hub"
    model_export = inspect_flat_export(
        model_source,
        repository=model["id"],
        revision=model["revision"],
        repo_type="model",
    )
    dataset_export = None
    if data_kind == "huggingface-dataset" and dataset_source is not None:
        dataset_export = inspect_flat_export(
            dataset_source,
            repository=data["repository"],
            revision=data["revision"],
            repo_type="dataset",
        )
    staged_model = stage_flat_export(model_export, hub_cache)
    staged_dataset = (
        stage_flat_export(dataset_export, hub_cache) if dataset_export else None
    )
    staged_dataset_file = (
        stage_huggingface_file(dataset_source, file_dataset_root, data)
        if data_kind == "huggingface-file"
        and dataset_source is not None
        and file_dataset_root is not None
        else None
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "target_id": target["target_id"],
        "registry_version": json.loads(
            (
                repo_root / "src/vllm_hust_benchmark/data/official_targets.json"
            ).read_text()
        )["registry_version"],
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": _sha256(spec_path),
        "hub_cache": str(hub_cache.resolve()),
        "model": staged_model,
        "dataset": staged_dataset,
        "dataset_file": staged_dataset_file,
    }
    _atomic_json(scratch_root / "hf-flat-cache-staging.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--dataset-source", type=Path)
    parser.add_argument("--file-dataset-root", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    os.umask(0o022)
    args = parse_args(argv)
    try:
        payload = stage_target(
            repo_root=args.repo_root,
            spec_path=args.spec,
            scratch_root=args.scratch_root,
            model_source=args.model_source,
            dataset_source=args.dataset_source,
            file_dataset_root=args.file_dataset_root,
        )
    except (OSError, KeyError, TypeError, ValueError, StagingFailure) as error:
        print(f"HF cache staging rejected: {error}", file=os.sys.stderr)
        return 2
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
