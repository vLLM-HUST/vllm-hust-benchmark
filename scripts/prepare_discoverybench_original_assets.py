"""Acquire the unchanged fixed official DiscoveryBench tree externally.

No Agent execution, data rewriting, environment installation or model calls.
All original data and publisher code stay outside Git; retain publisher terms.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from urllib.request import Request, urlopen

REPOSITORY = "allenai/discoverybench"
REVISION = "c31fcf011e070f021a5f5b906896d0821f6880e8"  # pragma: allowlist secret - public publisher revision
API = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
ARCHIVE = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{REVISION}"


def safe_path(root, name):
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or "\\" in name:
        raise ValueError("unsafe original archive path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("archive path escapes external root")
    return path


def blob_identity(path):
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def verify(root):
    tree = json.loads((root / "publisher-tree-private.json").read_text())
    if tree.get("truncated"):
        raise ValueError("publisher tree is incomplete")
    entries = [row for row in tree["tree"] if row["type"] == "blob"]
    if not entries:
        raise ValueError("empty publisher tree")
    for row in entries:
        if row["mode"] not in ("100644", "100755"):
            raise ValueError("unsupported linked original file")
        path = safe_path(root / "source", row["path"])
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != row["size"]
            or blob_identity(path) != row["sha"]
        ):
            raise ValueError("original Git blob/size mismatch: " + row["path"])
    expected = {row["path"] for row in entries}
    actual = {
        str(path.relative_to(root / "source").as_posix())
        for path in (root / "source").rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise ValueError("original tree file set differs from pinned publisher")
    return {
        "files_verified": len(entries),
        "original_bytes": sum(row["size"] for row in entries),
    }


def prepare(output, execute=False, verify_only=False):
    root = Path(output).resolve()
    result = {
        "source_id": "DISCOVERYBENCH",
        "status": "PLAN_ONLY",
        "repository": REPOSITORY,
        "revision": REVISION,
        "identity_note": "pragma: allowlist secret - public pinned publisher revision",
        "archive": ARCHIVE,
        "model_calls": 0,
        "benchmark_executed": False,
        "source_checkbox_eligible": False,
        "boundary": "Original asset preparation only. No license consolidation, native Agent/code execution or judge/scoring qualification. Publisher code/data terms retained separately; no raw redistribution.",
    }
    if verify_only:
        result.update(verify(root))
        result["status"] = "ORIGINAL_PINNED_TREE_VERIFIED_READ_ONLY_NOT_EXECUTED"
        return result
    if not execute:
        return result
    repository = Path(__file__).resolve().parents[1]
    if root.exists() or root.is_relative_to(repository) or not root.parent.is_dir():
        raise ValueError("fresh external destination with existing parent required")
    root.mkdir(mode=0o700)
    try:
        request = Request(
            API,
            headers={
                "User-Agent": "workload-native-preparation",
                "Accept": "application/vnd.github+json",
            },
        )
        with urlopen(request, timeout=60) as response:
            tree = json.load(response)
        if (
            tree.get("truncated")
            or sum(row.get("size", 0) for row in tree["tree"]) > 2 * 1024**3
        ):
            raise ValueError(
                "incomplete tree or unexpected asset size; no partial acquisition"
            )
        (root / "publisher-tree-private.json").write_text(json.dumps(tree))
        archive = root / "publisher-original.tar.gz"
        with (
            urlopen(
                Request(ARCHIVE, headers={"User-Agent": "workload-native-preparation"}),
                timeout=120,
            ) as response,
            archive.open("xb") as stream,
        ):
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                if stream.tell() > 2 * 1024**3:
                    raise ValueError("unexpected archive size")
        expected = {row["path"]: row for row in tree["tree"] if row["type"] == "blob"}
        source = root / "source"
        source.mkdir(mode=0o700)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                parts = PurePosixPath(member.name).parts
                if member.isdir():
                    continue
                name = "/".join(parts[1:])
                if not member.isfile() or name not in expected:
                    raise ValueError(
                        "unsupported or unexpected publisher archive member"
                    )
                if member.size != expected[name]["size"]:
                    raise ValueError(
                        "archive file size differs from publisher Git tree"
                    )
                destination = safe_path(source, name)
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with (
                    bundle.extractfile(member) as incoming,
                    destination.open("xb") as stream,
                ):
                    while chunk := incoming.read(1024 * 1024):
                        stream.write(chunk)
        result.update(verify(root))
        result["status"] = "ORIGINAL_PINNED_TREE_ACQUIRED_AND_VERIFIED_NOT_EXECUTED"
    except BaseException as error:
        result.update(
            status="FAILED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        (root / "preparation-public.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.execute, args.verify_only), indent=2))


if __name__ == "__main__":
    main()
