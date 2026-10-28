"""Prepare unchanged official DeepPlanning assets outside Git; never run a model."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from urllib.request import Request, urlopen

CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/deepplanning-original-assets.json"
)


def verify(path, entry):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked asset: {entry['path']}")
    if path.stat().st_size != entry["bytes"]:
        raise ValueError(f"original size mismatch: {entry['path']}")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        while chunk := stream.read(1024**2):
            digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"original digest mismatch: {entry['path']}")


def safe_path(root, name):
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or "\\" in name:
        raise ValueError("unsafe publisher member")
    path = root.joinpath(*parts.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("asset escapes output")
    return path


def fetch(url, target, entry):
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".partial")
    with (
        urlopen(
            Request(url, headers={"User-Agent": "workload-native-preparation"}),
            timeout=90,
        ) as response,
        temporary.open("xb") as stream,
    ):
        size = 0
        while chunk := response.read(1024**2):
            size += len(chunk)
            if size > entry["bytes"]:
                raise ValueError("publisher response exceeds fixed size")
            stream.write(chunk)
    verify(temporary, entry)
    temporary.rename(target)


def prepare(
    output,
    *,
    execute=False,
    verify_only=False,
    reuse_root=None,
    hf_endpoint="https://huggingface.co",
    config=None,
):
    manifest = config or json.loads(CONFIG.read_text("utf-8"))
    result = {
        "source_id": "DEEPPLANNING",
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "code_revision": manifest["code_revision"],
        "dataset_revision": manifest["dataset_revision"],
        "expected_files": len(manifest["files"]),
    }
    if execute and verify_only:
        raise ValueError("select execution or read-only verification")
    if not execute and not verify_only:
        return {**result, "status": "PLAN_ONLY"}
    output = Path(output).resolve()
    repo = CONFIG.parents[2]
    if output.is_relative_to(repo):
        raise ValueError("raw assets must stay outside repository")
    if hf_endpoint not in ("https://huggingface.co", "https://hf-mirror.com"):
        raise ValueError("select official endpoint or explicit same-version mirror")
    if verify_only:
        for entry in manifest["files"]:
            verify(safe_path(output, entry["path"]), entry)
        return {**result, "status": "READ_ONLY_ORIGINAL_ASSETS_VERIFIED"}
    if output.exists():
        raise ValueError("fresh output required; preserve prior attempts")
    reuse_root = Path(reuse_root).resolve() if reuse_root else None
    if reuse_root:
        for entry in manifest["files"]:
            verify(safe_path(reuse_root, entry["path"]), entry)
    output.mkdir(parents=True, mode=0o700)
    try:
        if reuse_root:
            for entry in manifest["files"]:
                target = safe_path(output, entry["path"])
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(safe_path(reuse_root, entry["path"]), target)
        else:
            archive_info = manifest["code_archive"]
            archive_path = output / "original-publisher.tar.gz"
            fetch(
                archive_info["url"],
                archive_path,
                {**archive_info, "path": archive_path.name},
            )
            expected = {
                entry["path"]: entry
                for entry in manifest["files"]
                if entry["path"].startswith("source/")
            }
            seen = set()
            with tarfile.open(archive_path, "r:gz") as archive:
                for member in archive:
                    parts = PurePosixPath(member.name)
                    if parts.is_absolute() or ".." in parts.parts:
                        raise ValueError("unsafe source archive")
                    name = "source/" + "/".join(parts.parts[1:])
                    if name not in expected:
                        continue
                    if (
                        name in seen
                        or not member.isfile()
                        or member.size != expected[name]["bytes"]
                    ):
                        raise ValueError("invalid or duplicate original source member")
                    target = safe_path(output, name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.extractfile(member) as stream,
                        target.open("xb") as destination,
                    ):
                        shutil.copyfileobj(stream, destination)
                    verify(target, expected[name])
                    seen.add(name)
            if seen != set(expected):
                raise ValueError("publisher archive lacks required original files")
            for entry in manifest["files"]:
                if entry["path"].startswith("databases/"):
                    url = f"{hf_endpoint}/datasets/{manifest['dataset_repository']}/resolve/{manifest['dataset_revision']}/{Path(entry['path']).name}"
                    fetch(url, safe_path(output, entry["path"]), entry)
        for entry in manifest["files"]:
            verify(safe_path(output, entry["path"]), entry)
        result["status"] = "ORIGINAL_ASSETS_VERIFIED_NATIVE_EXECUTION_PENDING"
    except Exception as error:
        result.update(
            status="PREPARATION_FAILED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        (output / "preparation-public.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reuse-root")
    parser.add_argument(
        "--hf-endpoint",
        default="https://huggingface.co",
        choices=("https://huggingface.co", "https://hf-mirror.com"),
    )
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
