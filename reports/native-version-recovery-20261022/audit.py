"""Recover package identities from exact archived version receipts, not answers."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile


SOURCES = (
    (
        "mmlu-pro",
        "quality-and-independent-performance",
        "reports/mmlu-pro-qwen35-b0-precheck-20261012/raw/run-evidence.tar.gz",
        "receipt/versions.txt",
    ),
    (
        "frontierscience",
        "old-native-server",
        "reports/frontierscience-qwen35-b0-precheck-20261013/selected-evidence.tar.gz",
        "server/receipt/versions.txt",
    ),
    (
        "hle-verified",
        "gold-quality",
        "reports/hle-verified-qwen35-20261015/gold-evidence.tar.gz",
        "server/versions.txt",
    ),
    (
        "hle-verified",
        "revision-quality",
        "reports/hle-verified-qwen35-20261015/revision-evidence.tar.gz",
        "server/versions.txt",
    ),
    (
        "hle-verified",
        "independent-performance",
        "reports/hle-verified-qwen35-20261015/bench-evidence.tar.gz",
        "server-startup/versions.txt",
    ),
)
EXPECTED = {
    "vllm": "0.28.1.post1.dev143+gf18cf803c.empty",
    "vllm_ascend": "0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909",
}


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_versions(payload):
    text = payload.decode("utf-8", errors="strict")
    result = {}
    for package in EXPECTED:
        values = re.findall(
            r"^" + re.escape(package) + r"\s+(\S+)(?:\s|$)", text, re.MULTILINE
        )
        if len(values) != 1:
            raise ValueError(
                f"Expected one package row for {package}, got {len(values)}"
            )
        result[package] = values[0]
    if result != EXPECTED:
        raise ValueError("Receipt does not match the recovered development runtime")
    return result


def inspect_source(root, dataset, scope, relative_path, member_name):
    path = root / relative_path
    with tarfile.open(path, "r:gz") as archive:
        matches = [item for item in archive.getmembers() if item.name == member_name]
        if len(matches) != 1 or not matches[0].isfile():
            raise ValueError(f"Expected exactly one regular member: {member_name}")
        if matches[0].size > 1024 * 1024:
            raise ValueError("Version receipt unexpectedly large")
        with archive.extractfile(matches[0]) as stream:
            payload = stream.read()
    return {
        "dataset": dataset,
        "scope": scope,
        "archive_path": relative_path,
        "archive_sha256": sha256_file(path),
        "member_path": member_name,
        "member_sha256": hashlib.sha256(payload).hexdigest(),
        "member_bytes": len(payload),
        "packages": parse_versions(payload),
        "runtime_family": "development-runtime",
        "designated_018_baseline": False,
    }


def build(roots):
    records = []
    for source in SOURCES:
        root = next((root for root in roots if (root / source[2]).is_file()), None)
        if root is None:
            raise FileNotFoundError(source[2])
        records.append(inspect_source(root, *source))
    return {
        "schema_version": 1,
        "kind": "derived-native-package-version-recovery",
        "repository": "vLLM-HUST/vllm-hust-benchmark",
        "method": "Parse exact package rows from the listed versions.txt members; hash archive bytes and member bytes. No original evidence is rewritten.",
        "limits": "Package identity only. Does not establish paired quality, isolated performance, source-tree cleanliness, or designated 0.18 acceptance. Scope follows the listed receipt; no transfer to other runs.",
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if bool(args.output) == bool(args.check):
        parser.error("Choose exactly one of --output or --check")
    result = build(args.repo)
    if args.check:
        if result != json.loads(args.check.read_text()):
            raise ValueError("Version mapping does not match original archive receipts")
        print("PASS: five archived receipts match the development-runtime mapping")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print("Wrote derived mapping for five receipts; originals unchanged")


if __name__ == "__main__":
    main()
