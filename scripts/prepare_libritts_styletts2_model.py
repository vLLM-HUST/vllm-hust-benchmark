"""Acquire pinned publisher LibriTTS model externally; plan by default."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/libritts-styletts2-model.json"


def target_path(root, item):
    relative = Path(item["path"])
    target = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not target.is_relative_to(root)
    ):
        raise ValueError("contained original model path required")
    return target


def verify(path, item):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]:
        raise ValueError("original regular model file with exact size required")
    digest = hashlib.sha256()
    git = hashlib.sha1(b"blob " + str(item["bytes"]).encode() + b"\0")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
            git.update(chunk)
    if (item.get("sha256") and digest.hexdigest() != item["sha256"]) or (
        item.get("git_blob") and git.hexdigest() != item["git_blob"]
    ):
        raise ValueError("publisher model identity differs; preserve failure")
    return digest.hexdigest()


def download_one(root, item, url, *, opener=urlopen):
    path = target_path(root, item)
    partial = path.with_name(path.name + ".partial")
    if path.exists() or partial.exists():
        raise ValueError("existing final/partial retained, never overwritten")
    if not 0 < item["bytes"] <= 1024**3 or not (
        item.get("sha256") or item.get("git_blob")
    ):
        raise ValueError("bounded publisher identity required")
    path.parent.mkdir(parents=True, exist_ok=True)
    with opener(
        Request(url, headers={"User-Agent": "pinned-LibriTTS-model"}), timeout=60
    ) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != item["bytes"]:
            raise ValueError("publisher response length differs")
        size = 0
        with partial.open("xb") as stream:
            while chunk := response.read(min(4 * 1024**2, item["bytes"] - size + 1)):
                size += len(chunk)
                if size > item["bytes"]:
                    raise ValueError("response exceeds exact original model budget")
                stream.write(chunk)
    digest = verify(partial, item)
    os.link(partial, path)
    partial.unlink()
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--hf-endpoint",
        choices=["https://huggingface.co", "https://hf-mirror.com"],
        default="https://huggingface.co",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true")
    action.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    pins = json.loads(PINS.read_bytes())
    report = {
        "source_id": "LIBRITTS",
        "status": "PLAN_ONLY",
        "model_repo": pins["repo_id"],
        "model_revision": pins["revision"],
        "files": pins["files"],
        "total_bytes": sum(item["bytes"] for item in pins["files"]),
        "transport_endpoint": args.hf_endpoint,
        "model_calls": 0,
        "checkpoint_deserialized": False,
        "source_checkbox_eligible": False,
    }
    if not args.download and not args.verify_only:
        print(json.dumps(report))
        return
    if args.output is None:
        parser.error("explicit external output required")
    root = args.output.resolve()
    if root.is_relative_to(ROOT) or root == root.parent:
        raise ValueError("model files must stay outside repository and filesystem root")
    os.umask(0o077)
    if args.verify_only:
        for item in pins["files"]:
            verify(target_path(root, item), item)
        report["status"] = (
            "ORIGINAL_MODEL_BYTES_VERIFIED_READ_ONLY_NOT_NATIVE_EXECUTION"
        )
        print(json.dumps(report))
        return
    if root.exists():
        raise ValueError("fresh task-owned output required")
    if shutil.disk_usage(root.parent).free < report["total_bytes"] + 64 * 1024**2:
        raise ValueError("insufficient space for pinned model and safety margin")
    root.mkdir(parents=True, mode=0o700, exist_ok=False)
    report.update(status="ACQUIRING_ORIGINAL_MODEL_BYTES", verified=[])
    receipt = root / "qualification-public.json"
    try:
        for item in pins["files"]:
            receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
            url = f"{args.hf_endpoint}/{pins['repo_id']}/resolve/{pins['revision']}/{item['path']}?download=true"
            digest = download_one(root, item, url)
            report["verified"].append(
                {"path": item["path"], "sha256": digest, "bytes": item["bytes"]}
            )
        report["status"] = (
            "ORIGINAL_MODEL_BYTES_ACQUIRED_NOT_RUNTIME_OR_NATIVE_QUALIFIED"
        )
    except BaseException as error:
        report.update(
            status="FAILED_RETAINED_NOT_NATIVE_EXECUTION",
            error_type=type(error).__name__,
            error=str(error)[:300],
        )
        raise
    finally:
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
