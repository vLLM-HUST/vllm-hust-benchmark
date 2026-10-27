"""Acquire a fixed upstream toolkit; never infer ChartQA execution from setup."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

REPOSITORY = "https://github.com/open-compass/VLMEvalKit.git"
# Public upstream Git object IDs, not credentials.
REVISION = "160e8a7d0cb1d4d739c1757038775cf2720d6895"  # pragma: allowlist secret
TREE = "417350395a289dd88538ac6cc031bf47c6e9f5d4"  # pragma: allowlist secret


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    ).stdout.strip()


def verify(root: Path, *, archive: bool = False) -> dict:
    root = root.resolve(strict=True)
    if not (root / ".git").is_dir():
        raise ValueError(
            "Expected a dedicated Git checkout, not an ancestor repository"
        )
    if archive:
        if git(root, "write-tree") != TREE:
            raise ValueError(
                "Archive content tree differs from the fixed upstream tree"
            )
        git(root, "diff", "--exit-code")
        if git(root, "ls-files", "--others"):
            raise ValueError("Toolkit has untracked files")
    else:
        if git(root, "rev-parse", "HEAD") != REVISION:
            raise ValueError(
                "Toolkit revision differs from the fixed upstream revision"
            )
        if git(root, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(
                "Toolkit has changed or untracked files; preserve and inspect it"
            )
    git(root, "fsck", "--no-reflogs")
    paths = git(root, "ls-files").splitlines()
    required = {
        "run.py",
        "vlmeval/dataset/image_vqa.py",
        "vlmeval/dataset/utils/vqa_eval.py",
    }
    if not required.issubset(paths):
        raise ValueError("Required original prompt/scorer/runner files are missing")
    return {
        "source_id": "CHARTQA",
        "status": "PINNED_TOOLKIT_SOURCE_VERIFIED_NOT_NATIVE_EXECUTION",
        "repository": REPOSITORY,
        "revision": REVISION,
        "tracked_files": len(paths),
        "git_tree": git(root, "write-tree")
        if archive
        else git(root, "rev-parse", "HEAD^{tree}"),
        "transport": "official_codeload_archive_exact_git_tree"
        if archive
        else "official_git_fetch",
        "model_calls": 0,
        "native_benchmark_verified": False,
    }


def acquire_archive(
    root: Path,
    *,
    repository: str = "open-compass/VLMEvalKit",
    revision: str = REVISION,
    tree: str = TREE,
    gitlinks: tuple[tuple[str, str], ...] = (),
) -> None:
    """Alternative official transport; verify every byte/mode via the original Git tree."""
    url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
    prefix = f"{repository.split('/')[-1]}-{revision}/"
    with tempfile.TemporaryFile(dir=root.parent) as payload:
        with urllib.request.urlopen(url, timeout=120) as response:
            size = 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > 256 * 1024 * 1024:
                    raise ValueError("Source archive exceeds 256 MiB safety budget")
                payload.write(chunk)
        payload.seek(0)
        with tarfile.open(fileobj=payload, mode="r:gz") as source:
            members = source.getmembers()
            if sum(item.size for item in members) > 1024**3:
                raise ValueError("Expanded source exceeds 1 GiB safety budget")
            seen = set()
            for item in members:
                if item.name.rstrip("/") == prefix.rstrip("/") and item.isdir():
                    continue
                if not item.name.startswith(prefix):
                    raise ValueError("Unexpected archive root")
                relative = item.name[len(prefix) :]
                if (
                    not relative
                    or relative.startswith("/")
                    or ".." in Path(relative).parts
                    or ".git" in Path(relative).parts
                    or ":" in relative
                    or "\\" in relative
                    or Path(relative).is_absolute()
                    or not (item.isdir() or item.isfile())
                    or relative in seen
                ):
                    raise ValueError("Unsafe or duplicate source member")
                seen.add(relative)
                target = root / relative
                if item.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        source.extractfile(item) as contents,
                        target.open("xb") as output,
                    ):
                        while chunk := contents.read(1024 * 1024):
                            output.write(chunk)
                    target.chmod(0o755 if item.mode & 0o111 else 0o644)
    git(root, "init")
    git(root, "-c", "core.autocrlf=false", "add", "--force", "--all")
    for path, commit in gitlinks:
        git(root, "update-index", "--add", "--cacheinfo", f"160000,{commit},{path}")
    if git(root, "write-tree") != tree:
        raise ValueError(
            "Downloaded archive does not reproduce the fixed original Git tree"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Official codeload transport; exact fixed Git tree verification",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    if args.verify_only:
        result = verify(root, archive=args.archive)
    elif args.execute:
        if root.exists():
            raise ValueError(
                "Output already exists; use --verify-only, never overwrite"
            )
        root.mkdir(parents=True, exist_ok=False)
        if args.archive:
            acquire_archive(root)
        else:
            git(root, "init")
            git(root, "fetch", "--depth=1", REPOSITORY, REVISION)
            git(root, "checkout", "--detach", "FETCH_HEAD")
        result = verify(root, archive=args.archive)
    else:
        result = {
            "status": "PLAN_ONLY_NO_DOWNLOAD_OR_MODEL_CALL",
            "repository": REPOSITORY,
            "revision": REVISION,
            "output": str(root),
            "remaining": "Isolated dependencies, original dataset/protocol selection, NPU inference, original scoring and research handoff",
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
