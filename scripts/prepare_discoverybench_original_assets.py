"""Acquire the unchanged fixed official DiscoveryBench tree externally.

No Agent execution, data rewriting, environment installation or model calls.
All original data and publisher code stay outside Git; retain publisher terms.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from urllib.request import Request, urlopen
from urllib.parse import quote

REPOSITORY = "allenai/discoverybench"
REVISION = "c31fcf011e070f021a5f5b906896d0821f6880e8"  # pragma: allowlist secret - public publisher revision
API = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
ARCHIVE = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{REVISION}"
HF_REVISION = "e54ec033049d3a0fd95d3c746919cc8c01c25781"  # pragma: allowlist secret - public official dataset revision
HF_INDEX_FILES = {
    "README.md": (
        2875,
        "db19cd8e73681355c543b6401ce1599037e4bd1d",  # pragma: allowlist secret - original public Git blob
    ),
    "train_relevant.csv": (
        111066,
        "db688984280011b36a62bb46832391adf11d0cdb",  # pragma: allowlist secret - original public Git blob
    ),
    "test_relevant.csv": (
        2297184,
        "2ec6c4da25f26e72c28a42a0305e783f93357686",  # pragma: allowlist secret - original public Git blob
    ),
}
HF_ANSWER_FILES = {
    "answer_key/answer_key_real.csv": (
        57833,
        "b991286ce5c6efecfd15e2d2c93e3075e04b8920",  # pragma: allowlist secret - original public Git blob
    ),
    "answer_key/answer_key_synth.csv": (
        61658,
        "2fabeb1557e6aa0d8d07a3206ed70f3ec6179ca0",  # pragma: allowlist secret - original public Git blob
    ),
}
HF_NATIVE_REAL_PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/discoverybench-native-real-pins.json"
)


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


def prepare_publisher_index(
    output,
    execute=False,
    verify_only=False,
    hf_endpoint="https://huggingface.co",
    answer_keys=False,
    reuse_assets=None,
    native_real=False,
):
    if hf_endpoint not in ("https://huggingface.co", "https://hf-mirror.com"):
        raise ValueError("only fixed public HF transports are supported")
    root = Path(output).resolve()
    files = HF_ANSWER_FILES if answer_keys else HF_INDEX_FILES
    if native_real:
        if answer_keys:
            raise ValueError("native data and answer-key profiles are separate")
        pins = json.loads(HF_NATIVE_REAL_PINS.read_text("utf-8"))
        if pins.get(
            "schema_version"
        ) != "discoverybench-original-native-real/v1" or not pins.get("files"):
            raise ValueError("missing or invalid complete native-data pins")
        if pins["revision"] != HF_REVISION:
            raise ValueError(
                "native data pin revision differs from official reference release"
            )
        files = {}
        for row in pins["files"]:
            if (
                not row["path"].startswith("discoverybench/real/")
                or not isinstance(row["bytes"], int)
                or row["bytes"] < 0
                or not re.fullmatch(r"[a-f0-9]{40}", row["git_blob"])
            ):
                raise ValueError("invalid pinned native data identity")
            if row["path"] in files:
                raise ValueError("duplicate pinned native data path")
            files[row["path"]] = (row["bytes"], row["git_blob"])
    result = {
        "source_id": "DISCOVERYBENCH",
        "release": "publisher-native-real"
        if native_real
        else ("publisher-answer-keys" if answer_keys else "publisher-task-index"),
        "publisher": "https://huggingface.co/datasets/allenai/discoverybench",
        "revision": HF_REVISION,
        "identity_note": "pragma: allowlist secret - fixed official dataset revision",
        "status": "PLAN_ONLY",
        "transport_endpoint": hf_endpoint,
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "benchmark_executed": False,
        "boundary": "Selected pinned original publisher assets only, not native Agent/scoring or proof of task/answer joins. Native-real is a complete pinned real-data subtree, not synthetic data or an assertion that GitHub/CSV exports are interchangeable. Reference answers must not become model input.",
    }
    if not execute and not verify_only:
        return result
    if execute:
        if reuse_assets is not None and not Path(reuse_assets).is_dir():
            raise ValueError("explicit reuse asset directory does not exist")
        if (
            root.exists()
            or root.is_relative_to(Path(__file__).resolve().parents[1])
            or not root.parent.is_dir()
        ):
            raise ValueError("fresh external publisher-index destination required")
        root.mkdir(mode=0o700)
    try:
        counts = {}
        reused = []
        rejected_reuse = []
        parse_errors = []
        for name, (size, identity) in files.items():
            path = safe_path(root, name)
            if execute:
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                previous = Path(reuse_assets) / name if reuse_assets else None
                if (
                    previous is not None
                    and previous.is_file()
                    and (
                        previous.stat().st_size != size
                        or blob_identity(previous) != identity
                    )
                ):
                    if not native_real:
                        raise ValueError(
                            "reused publisher file differs from fixed original"
                        )
                    rejected_reuse.append(name)
                    previous = None
                if previous is not None and previous.is_file():
                    if (
                        previous.stat().st_size != size
                        or blob_identity(previous) != identity
                    ):
                        raise ValueError(
                            "reused publisher file differs from fixed original"
                        )
                    with previous.open("rb") as source, path.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
                    reused.append(name)
                else:
                    download_publisher_file(path, name, size, hf_endpoint)
            if path.stat().st_size != size or blob_identity(path) != identity:
                raise ValueError("fixed original publisher index blob/size mismatch")
            if name.endswith(".csv") and not native_real:
                try:
                    with path.open(encoding="utf-8", newline="") as stream:
                        reader = csv.DictReader(stream)
                        fields = reader.fieldnames
                        rows = sum(1 for _ in reader)
                    counts[name] = {"original_rows": rows, "original_columns": fields}
                except (UnicodeDecodeError, csv.Error) as error:
                    parse_errors.append(
                        {
                            "file": name,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                    )
        result.update(
            status="ORIGINAL_PUBLISHER_NATIVE_REAL_FILES_VERIFIED_NOT_EXECUTED"
            if native_real
            else (
                "ORIGINAL_PUBLISHER_FILES_VERIFIED_WITH_PARSE_ERRORS_NOT_NATIVE_EXECUTION"
                if parse_errors
                else "ORIGINAL_PUBLISHER_INDEX_VERIFIED_NOT_NATIVE_EXECUTION"
            ),
            original_csv_structure=counts,
            files_verified=len(files),
            csv_parse_errors=parse_errors,
            reused_original_files=reused,
            rejected_reuse_files=rejected_reuse,
            downloaded_original_files=[name for name in files if name not in reused]
            if execute
            else [],
        )
    except BaseException as error:
        result.update(
            status="FAILED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        if execute:
            (root / "preparation-public.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
    return result


def download_publisher_file(path, name, size, hf_endpoint):
    url = f"{hf_endpoint}/datasets/allenai/discoverybench/resolve/{HF_REVISION}/{quote(name, safe='/')}"
    with (
        urlopen(
            Request(url, headers={"User-Agent": "workload-native-preparation"}),
            timeout=120,
        ) as response,
        path.open("xb") as stream,
    ):
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
            if stream.tell() > size:
                raise ValueError("official index exceeds fixed original size")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reuse-assets",
        type=Path,
        help="Copy only identity-verified existing publisher files to a fresh external destination; fetch missing files only",
    )
    parser.add_argument(
        "--release",
        choices=(
            "github-tree",
            "publisher-task-index",
            "publisher-answer-keys",
            "publisher-native-real",
        ),
        default="github-tree",
    )
    parser.add_argument(
        "--hf-endpoint",
        choices=("https://huggingface.co", "https://hf-mirror.com"),
        default="https://huggingface.co",
        help="Explicit transport fallback only; identical official revision/blob/size required, no credentials",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.release == "github-tree":
        if (
            args.hf_endpoint != "https://huggingface.co"
            or args.reuse_assets is not None
        ):
            parser.error("--hf-endpoint and --reuse-assets apply only to publisher releases")
        result = prepare(args.output, args.execute, args.verify_only)
    else:
        result = prepare_publisher_index(
            args.output,
            args.execute,
            args.verify_only,
            args.hf_endpoint,
            args.release == "publisher-answer-keys",
            args.reuse_assets,
            args.release == "publisher-native-real",
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
