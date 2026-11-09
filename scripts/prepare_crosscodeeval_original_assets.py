"""Pinned author code/data preparation only; no prompts, model calls or scoring."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from prepare_xsum_original_assets import bound_path, verify

PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/crosscodeeval-author-pins.json"
)


def git_lfs_pointer(item):
    if "git_lfs" not in item:
        return None
    spec = item["git_lfs"]
    if not isinstance(spec, dict) or set(spec) != {"pointer_bytes", "oid_sha256"}:
        raise ValueError("explicit original Git LFS pointer identity required")
    oid = spec["oid_sha256"]
    if (
        not isinstance(oid, str)
        or len(oid) != 64
        or any(char not in "0123456789abcdef" for char in oid)
        or oid != item["sha256"]
        or type(item["bytes"]) is not int
        or item["bytes"] < 0
    ):
        raise ValueError("Git LFS OID must bind the unchanged payload SHA256")
    pointer = (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {item['bytes']}\n"
    ).encode()
    if len(pointer) != spec["pointer_bytes"]:
        raise ValueError("original Git LFS pointer size differs")
    return pointer


def git_blob_for_payload(raw, item):
    pointer = git_lfs_pointer(item)
    if pointer is not None and (
        len(raw) != item["bytes"] or hashlib.sha256(raw).hexdigest() != item["sha256"]
    ):
        raise ValueError("expanded Git LFS payload differs from the original OID")
    value = raw if pointer is None else pointer
    return hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()


def verify_git_blob(path, item):
    verify(path, item)
    pointer = git_lfs_pointer(item)
    if pointer is not None:
        digest = hashlib.sha1(b"blob " + str(len(pointer)).encode() + b"\0" + pointer)
        if digest.hexdigest() != item["git_blob"]:
            raise ValueError("original publisher Git LFS pointer identity differs")
        return
    prefix = b"blob " + str(item["bytes"]).encode() + b"\0"
    digest = hashlib.sha1(prefix)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != item["git_blob"]:
        raise ValueError("original publisher Git blob identity differs")


def acquire(path, item):
    if "git_lfs" in item:
        raise ValueError(
            "expanded Git LFS payload requires the original publisher archive,not a pointer-only Git blob download"
        )
    if path.exists() or path.with_name(path.name + ".partial").exists():
        raise ValueError("existing final/partial must be inspected, never overwritten")
    limit = item["bytes"] * 2 + 65536
    with urlopen(
        Request(item["url"], headers={"User-Agent": "CrossCodeEval-pinned-assets"}),
        timeout=90,
    ) as response:
        encoded = response.read(limit + 1)
    if len(encoded) > limit:
        raise ValueError("publisher Git blob response exceeds pinned size budget")
    payload = json.loads(encoded)
    if payload.get("sha") != item["git_blob"] or payload.get("encoding") != "base64":
        raise ValueError("pinned publisher base64 Git blob required")
    raw = base64.b64decode("".join(payload["content"].split()), validate=True)
    if len(raw) != item["bytes"]:
        raise ValueError("decoded original asset size differs")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as stream:
        stream.write(raw)
    verify_git_blob(partial, item)
    os.link(partial, path)
    partial.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--data-archive", type=Path, required=True)
    parser.add_argument(
        "--action", choices=["plan", "prepare", "verify"], default="plan"
    )
    parser.add_argument("--max-bytes", type=int)
    args = parser.parse_args()
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    jobs = [
        (bound_path(args.author_root, "", item), item) for item in pins["author_files"]
    ]
    jobs.append((args.data_archive.resolve(), pins["dataset"]))
    total = sum(item["bytes"] for _, item in jobs)
    if args.action == "prepare" and (args.max_bytes is None or args.max_bytes < total):
        raise ValueError(
            "explicit max-bytes must cover the entire pinned code/data selection"
        )
    verified = 0
    if args.action != "plan":
        for path, item in jobs:
            if args.action == "prepare" and not path.exists():
                acquire(path, item)
            verify_git_blob(path, item)
            verified += 1
    print(
        json.dumps(
            {
                "source_id": "CROSSCODEEVAL",
                "revision": pins["revision"],
                "action": args.action,
                "selected_assets": len(jobs),
                "selected_bytes": total,
                "verified_assets": verified,
                "model_calls": 0,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
