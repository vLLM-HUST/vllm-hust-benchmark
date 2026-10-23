#!/usr/bin/env python3
"""Verify the independent H2 ownership review against frozen Git objects."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "independent-oracle-review.json"
RECEIPT = ROOT / "independent-oracle-receipt.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: str, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", repo, *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def git_result(repo: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", repo, *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> int:
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt["review_sha256"] != sha256(REVIEW):
        fail("review SHA-256 differs from receipt")
    if receipt["verifier_sha256"] != sha256(Path(__file__).resolve()):
        fail("verifier SHA-256 differs from receipt")
    campaign = Path(review["campaign_binding"]["path"])
    if sha256(campaign) != review["campaign_binding"]["sha256"]:
        fail("campaign SHA-256 differs from frozen binding")
    campaign_data = json.loads(campaign.read_text(encoding="utf-8"))
    expected_campaign = {
        "core": review["sources"]["host_core"]["commit"],
        "ascend": review["sources"]["host_ascend"]["commit"],
        "lifecycle_profiler": review["sources"]["lifecycle_profiler"]["commit"],
        "kvcompress": review["sources"]["kvcompress"]["commit"],
    }
    observed_campaign = {
        "core": campaign_data["host"]["core"]["commit"],
        "ascend": campaign_data["host"]["ascend"]["commit"],
        **{member["id"]: member["commit"] for member in campaign_data["members"]},
    }
    if observed_campaign != expected_campaign:
        fail("campaign source identities differ from review")
    if [row["order_id"] for row in campaign_data["installation_orders"]] != [
        "profiler-then-kvcompress",
        "kvcompress-then-profiler",
    ]:
        fail("campaign installation orders changed")
    for source_name, source in review["sources"].items():
        repo = source["repo"]
        commit = source["commit"]
        if git(repo, "rev-parse", f"{commit}^{{commit}}") != commit:
            fail(f"{source_name}: commit cannot be resolved exactly")
        if git(repo, "rev-parse", f"{commit}^{{tree}}") != source["tree"]:
            fail(f"{source_name}: tree differs")
        for file_name, blob in source["files"].items():
            if git(repo, "rev-parse", f"{commit}:{file_name}") != blob:
                fail(f"{source_name}: blob differs for {file_name}")
        for file_name in source.get("absent_paths", []):
            if git_result(repo, "cat-file", "-e", f"{commit}:{file_name}").returncode == 0:
                fail(f"{source_name}: expected path is present: {file_name}")
    for ownership in review["ownership"]:
        for evidence in ownership["evidence"]:
            source = review["sources"][evidence["source"]]
            file_name = evidence["file"]
            if file_name not in source["files"]:
                fail(f"evidence cites unbound file: {evidence['source']}:{file_name}")
            content = git(
                source["repo"], "show", f"{source['commit']}:{file_name}"
            )
            line_count = len(content.splitlines())
            cited_lines = [int(value) for value in re.findall(r"\d+", evidence["lines"])]
            if not cited_lines or min(cited_lines) < 1 or max(cited_lines) > line_count:
                fail(
                    f"evidence line range invalid: {evidence['source']}:{file_name}:"
                    f"{evidence['lines']} (file has {line_count} lines)"
                )
    if review["verdict"] not in {"admit", "reject", "unknown"}:
        fail("verdict is outside closed vocabulary")
    if review["verdict"] != "admit":
        fail("receipt was issued for a different verdict")
    print(
        "PASS independent-oracle-review "
        f"verdict={review['verdict']} review_sha256={sha256(REVIEW)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
