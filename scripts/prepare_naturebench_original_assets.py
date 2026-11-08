"""Prepare unchanged official NatureBench CPU task packages; no Agent execution.

Default is an offline plan. Uses the unmodified author download-only CLI and the
official huggingface_hub0.36 HTTP-backend extension for explicit mirror pagination.
All original files are independently checked against pinned publisher identities.
"""

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import sys
import tempfile
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from prepare_jsonschemabench_author_source import verify_source

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/workload-native/naturebench-original-cpu-files.json"
RELEASE = ROOT / "configs/workload-native/naturebench-original-release.json"
AUTHOR = ROOT / "configs/workload-native/naturebench-author-source.json"
ENDPOINTS = {"https://huggingface.co", "https://hf-mirror.com"}


def load_profile():
    profile = json.loads(PROFILE.read_text("utf-8"))
    release = json.loads(RELEASE.read_text("utf-8"))
    if (
        profile["schema_version"] != "naturebench-original-task-files/v1"
        or profile["repo_id"] != release["dataset_repository"]
        or profile["revision"] != release["dataset_revision"]
        or len(profile["files"]) != 576
        or sum(item["bytes"] for item in profile["files"]) != 2754532128
    ):
        raise ValueError("fixed original CPU profile identity or inventory differs")
    seen = set()
    for item in profile["files"]:
        path = PurePosixPath(item["path"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in item["path"]
            or len(path.parts) < 3
            or path.parts[0] != "tasks"
            or path.parts[1] not in release["official_cpu_tasks"]
            or item["path"] in seen
            or type(item["bytes"]) is not int
            or item["bytes"] < 0
            or ("sha256" in item) == ("git_blob" in item)
            or not re.fullmatch(
                "[0-9a-f]{64}" if "sha256" in item else "[0-9a-f]{40}",
                item.get("sha256", item.get("git_blob", "")),
            )
        ):
            raise ValueError(
                "distinct contained original task file identities required"
            )
        seen.add(item["path"])
    for task in release["cpu_task_metadata"]:
        rows = [
            item
            for item in profile["files"]
            if item["path"].startswith(f"tasks/{task['task']}/")
        ]
        if (
            len(rows) != task["files"]
            or sum(row["bytes"] for row in rows) != task["bytes"]
        ):
            raise ValueError("original per-task file inventory differs")
    return profile


def pagination_url(url, endpoint, revision):
    """Change only a canonical same-revision tree Link's explicit transport host."""
    if endpoint not in ENDPOINTS or not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("explicit supported endpoint and pinned revision required")
    parsed = urlsplit(url)
    prefix = "/api/datasets/FrontisAI/NatureBench/tree/"
    if parsed.hostname != "huggingface.co" or not parsed.path.startswith(prefix):
        return url
    if (
        parsed.scheme != "https"
        or parsed.netloc != "huggingface.co"
        or parsed.path.removeprefix(prefix).split("/", 1)[0] != revision
        or parsed.fragment
    ):
        raise ValueError("pagination must retain exact original dataset revision")
    return urlunsplit(
        ("https", urlsplit(endpoint).netloc, parsed.path, parsed.query, "")
    )


def backend_factory(endpoint, revision):
    import requests

    class OriginalPaginationAdapter(requests.adapters.HTTPAdapter):
        def send(self, request, **kwargs):
            request.url = pagination_url(request.url, endpoint, revision)
            if kwargs.get("timeout") is None:
                kwargs["timeout"] = (15, 60)
            return super().send(request, **kwargs)

    session = requests.Session()
    session.mount("https://", OriginalPaginationAdapter())
    return session


def verify_files(data_root, profile, *, complete=True):
    data_root = Path(data_root).resolve(strict=True)
    expected = {item["path"] for item in profile["files"]}
    actual = {
        path.relative_to(data_root).as_posix()
        for path in (data_root / "tasks").rglob("*")
        if path.is_file()
    }
    if (complete and actual != expected) or not actual <= expected:
        raise ValueError("original task files missing or additional task files present")
    for item in profile["files"]:
        if item["path"] not in actual:
            continue
        path = (data_root / item["path"]).resolve(strict=True)
        if not path.is_relative_to(data_root) or path.stat().st_size != item["bytes"]:
            raise ValueError("contained original file size differs")
        digest = hashlib.sha256() if "sha256" in item else hashlib.sha1()
        if "git_blob" in item:
            digest.update(b"blob " + str(item["bytes"]).encode() + b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024**2), b""):
                digest.update(block)
        if digest.hexdigest() != item.get("sha256", item.get("git_blob")):
            raise ValueError("original publisher file digest differs")
    return len(actual)


def fetch_publisher_file(data_root, cache_root, item, profile, endpoint):
    """Fetch one entire original object, never a task/content subset."""
    if endpoint not in ENDPOINTS:
        raise ValueError("explicit supported publisher transport required")
    data_root, cache_root = Path(data_root).resolve(), Path(cache_root).resolve()
    target = (data_root / item["path"]).resolve()
    if not target.is_relative_to(data_root) or target.exists():
        raise ValueError("fresh contained original target required")
    url = (
        f"{endpoint}/datasets/{profile['repo_id']}/resolve/{profile['revision']}/"
        f"{quote(item['path'], safe='/')}?download=true"
    )
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "original-task-asset-preparation",
    }
    if item["bytes"]:
        headers["Range"] = f"bytes=0-{item['bytes'] - 1}"
    digest = hashlib.sha256() if "sha256" in item else hashlib.sha1()
    if "git_blob" in item:
        digest.update(b"blob " + str(item["bytes"]).encode() + b"\0")
    size = 0
    temporary_root = data_root / ".original-asset-tmp"
    temporary_root.mkdir(mode=0o700, exist_ok=True)
    if not temporary_root.resolve().is_relative_to(data_root):
        raise ValueError("contained original temporary directory required")
    with tempfile.NamedTemporaryFile(
        dir=temporary_root, prefix="original-object-", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            for block in iter(lambda: response.read(4 * 1024**2), b""):
                size += len(block)
                if size > item["bytes"]:
                    raise ValueError("original object exceeds declared byte budget")
                digest.update(block)
                temporary.write(block)
    if size != item["bytes"] or digest.hexdigest() != item.get(
        "sha256", item.get("git_blob")
    ):
        raise ValueError(
            "original publisher object size/digest differs; failed temporary retained"
        )
    target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    temporary_path.rename(target)


def download(
    author_root,
    data_root,
    cache_root,
    profile,
    endpoint,
    *,
    resume=False,
    transport="original-cli",
):
    if transport not in {"original-cli", "publisher-files"}:
        raise ValueError("explicit supported original asset transport required")
    if transport == "original-cli" and metadata.version("huggingface-hub") != "0.36.0":
        raise ValueError("use isolated huggingface-hub==0.36.0 preparation client")
    author_root = verify_source(author_root, AUTHOR)
    data_root, cache_root = Path(data_root).resolve(), Path(cache_root).resolve()
    paths = [author_root, data_root, cache_root]
    if any(
        a == b or a.is_relative_to(b)
        for i, a in enumerate(paths)
        for j, b in enumerate(paths)
        if i != j
    ):
        raise ValueError("original source, fresh data and cache must be separate")
    if resume:
        if not data_root.is_dir() or not cache_root.is_dir():
            raise ValueError(
                "explicit resume requires existing original data/cache roots"
            )
    elif data_root.exists() or cache_root.exists():
        raise ValueError("fresh roots or explicit verified resume required")
    if shutil.disk_usage(data_root.parent).free < 100 * 1024**3:
        raise ValueError(
            "at least100GiB free required before original task preparation"
        )
    if not resume:
        data_root.mkdir(mode=0o700)
        cache_root.mkdir(mode=0o700)
    (cache_root / "tmp").mkdir(mode=0o700, exist_ok=True)
    from filelock import FileLock

    with FileLock(str(cache_root / ".naturebench-download.lock"), timeout=0):
        if resume:
            verify_files(data_root, profile, complete=False)
        if transport == "publisher-files":
            for item in profile["files"]:
                if not (data_root / item["path"]).exists():
                    fetch_publisher_file(data_root, cache_root, item, profile, endpoint)
            return verify_files(data_root, profile)
        return _download_original(author_root, data_root, cache_root, profile, endpoint)


def _download_original(author_root, data_root, cache_root, profile, endpoint):
    changes = {
        "HF_ENDPOINT": endpoint,
        "HF_HOME": str(cache_root),
        "HF_HUB_CACHE": str(cache_root / "hub"),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_ETAG_TIMEOUT": "15",
        "HF_HUB_DOWNLOAD_TIMEOUT": "60",
        "TMPDIR": str(cache_root / "tmp"),
        "HF_TOKEN": None,
        "HUGGING_FACE_HUB_TOKEN": None,
        "HF_API_TOKEN": None,
        "OPENAI_API_KEY": None,
    }
    old_env = {key: os.environ.get(key) for key in changes}
    argv, directory, search_path = sys.argv, Path.cwd(), list(sys.path)
    try:
        for key, value in changes.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from huggingface_hub import configure_http_backend

        configure_http_backend(lambda: backend_factory(endpoint, profile["revision"]))
        sys.path.insert(0, str(author_root))
        os.chdir(author_root)
        sys.argv = [
            str(author_root / "run_naturebench.py"),
            "--config",
            str(author_root / "config.example.yaml"),
            "--dataset-id",
            profile["repo_id"],
            "--dataset-revision",
            profile["revision"],
            "--tasks",
            "cpu",
            "--data-dir",
            str(data_root),
            "--download-only",
        ]
        try:
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise ValueError("original author download-only failed") from error
    finally:
        sys.argv, sys.path = argv, search_path
        os.chdir(directory)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if "huggingface_hub" in sys.modules:
            from huggingface_hub import configure_http_backend

            configure_http_backend()
    return verify_files(data_root, profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["plan", "download", "verify"], default="plan"
    )
    parser.add_argument("--author-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument(
        "--hf-endpoint", choices=sorted(ENDPOINTS), default="https://huggingface.co"
    )
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument(
        "--transport",
        choices=["original-cli", "publisher-files"],
        default="original-cli",
        help="Prefer original CLI; publisher-files fetches the exact same entire pinned objects",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Verify completed original files before resuming the same snapshot",
    )
    args = parser.parse_args()
    profile = load_profile()
    total = sum(item["bytes"] for item in profile["files"])
    result = {
        "source_id": "NATUREBENCH",
        "status": "OFFLINE_ORIGINAL_CPU_ASSET_PLAN",
        "revision": profile["revision"],
        "files": len(profile["files"]),
        "bytes": total,
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "boundary": "Original CPU task assets only, not scientific Agent/evaluator/judge execution, whole90-task coverage or ARM/NPU runtime qualification",
    }
    if args.mode != "plan" and args.data_root is None:
        parser.error("explicit external original data root required")
    if args.mode == "download":
        if (
            args.author_root is None
            or args.cache_root is None
            or args.max_bytes < total
        ):
            parser.error(
                "original source, fresh external cache and sufficient explicit byte budget required"
            )
        result["verified_files"] = download(
            args.author_root,
            args.data_root,
            args.cache_root,
            profile,
            args.hf_endpoint,
            resume=args.resume,
            transport=args.transport,
        )
        result["status"] = "ORIGINAL_CPU_TASK_ASSETS_VERIFIED_NOT_NATIVE_DELIVERY"
    elif args.mode == "verify":
        result["verified_files"] = verify_files(args.data_root, profile)
        result["status"] = (
            "ORIGINAL_CPU_TASK_ASSETS_VERIFIED_READ_ONLY_NOT_NATIVE_DELIVERY"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
