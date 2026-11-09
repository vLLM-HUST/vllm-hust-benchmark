"""Verify public Release bytes and TraceLoom members without modifying releases."""

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

REPOS = ("vLLM-HUST/vllm-hust-benchmark", "vLLM-HUST/vllm-hust-website")
TRACE_TAG = "evidence-traceloom-tp2-20261027-v1"


def pages(path):
    rows = []
    for page in range(1, 1000):
        headers = {"Accept": "application/vnd.github+json"}
        if token := os.environ.get("GH_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"
        url = f"https://api.github.com/{path}?per_page=100&page={page}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers), timeout=60
                ) as response:
                    batch = json.load(response)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        if not isinstance(batch, list):
            raise ValueError("Expected GitHub array")
        rows.extend(batch)
        if len(batch) < 100:
            return rows
    raise ValueError("Pagination limit exceeded")


def inventory():
    jobs, excluded = [], []
    for repo in REPOS:
        for release in pages(f"repos/{repo}/releases"):
            if release["draft"]:
                continue
            for asset in pages(f"repos/{repo}/releases/{release['id']}/assets"):
                row = dict(
                    repository=repo,
                    tag=release["tag_name"],
                    asset_id=asset["id"],
                    name=asset["name"],
                    bytes=asset["size"],
                    digest=asset.get("digest"),
                    url=asset["browser_download_url"],
                )
                if asset["state"] == "uploaded":
                    jobs.append(row)
                else:
                    excluded.append(dict(row, state=asset["state"]))
    return jobs, excluded


def check_stream(source, size, digest, destination=None):
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Missing or invalid SHA256")
    count, actual = 0, hashlib.sha256()
    while chunk := source.read(1024 * 1024):
        count += len(chunk)
        if count > size:
            raise ValueError("Asset exceeds declared size")
        actual.update(chunk)
        if destination is not None:
            destination.write(chunk)
    if count != size or "sha256:" + actual.hexdigest() != digest:
        raise ValueError(
            f"Identity mismatch: received {count}/{size} bytes, sha256:{actual.hexdigest()}"
        )
    return count


def download(row, destination=None):
    if not row["url"].startswith(
        f"https://github.com/{row['repository']}/releases/download/"
    ):
        raise ValueError("Unexpected asset destination")
    # Never send the API token to download or redirect hosts.
    with urllib.request.urlopen(row["url"], timeout=90) as response:
        return check_stream(response, row["bytes"], row["digest"], destination)


def check_members(source, expected):
    seen = set()
    with tarfile.open(fileobj=source, mode="r|") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name in seen
                or member.name not in expected
            ):
                raise ValueError(f"Unexpected member: {member.name}")
            record = expected[member.name]
            with archive.extractfile(member) as content:
                check_stream(content, record["size"], "sha256:" + record["sha256"])
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Missing archive members")
    return len(seen)


def verify(row, manifests):
    errors = []
    for attempt in range(3):
        try:
            count = 0
            key = row["repository"], row["tag"], row["name"]
            if key not in manifests:
                download(row)
            else:
                record = manifests[key]
                if (
                    record["bytes"] != row["bytes"]
                    or "sha256:" + record["sha256"] != row["digest"]
                ):
                    raise ValueError(
                        "Archive identity differs between API and member manifest"
                    )
                with tempfile.NamedTemporaryFile(suffix=".tar.zst") as archive:
                    download(row, archive)
                    archive.flush()
                    decoder = subprocess.Popen(
                        ["zstd", "-dc", archive.name], stdout=subprocess.PIPE
                    )
                    try:
                        count = check_members(decoder.stdout, record["members"])
                        while decoder.stdout.read(1024 * 1024):
                            pass
                        if decoder.wait() != 0:
                            raise ValueError("Decompression failed")
                    finally:
                        decoder.stdout.close()
                        if decoder.poll() is None:
                            decoder.kill()
                        decoder.wait()
            return dict(row, passed=True, verified_members=count, prior_errors=errors)
        except (OSError, ValueError, tarfile.TarError) as exc:
            errors.append(str(exc))
            if attempt < 2:
                time.sleep(2**attempt)
    return dict(row, passed=False, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jobs, excluded = inventory()
    if not jobs:
        raise ValueError("Empty public inventory")
    manifests = {}
    for row in jobs:
        if row["tag"] == TRACE_TAG and row["name"] == "MANIFEST.json":
            body = io.BytesIO()
            download(row, body)
            for record in json.loads(body.getvalue())["archives"]:
                manifests[row["repository"], row["tag"], record["name"]] = record
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for future in concurrent.futures.as_completed(
            [pool.submit(verify, row, manifests) for row in jobs]
        ):
            results.append(future.result())
            if len(results) % 50 == 0:
                print(f"Checked {len(results)}/{len(jobs)} assets", flush=True)
    results.sort(key=lambda r: (r["repository"], r["tag"], r["name"]))
    summary = dict(
        total=len(results),
        passed=sum(r["passed"] for r in results),
        verified_bytes=sum(r["bytes"] for r in results if r["passed"]),
        verified_members=sum(r.get("verified_members", 0) for r in results),
        excluded_nonuploaded=len(excluded),
    )
    report = dict(
        schema="public-release-custody/v1",
        checked_at=datetime.now(timezone.utc).isoformat(),
        summary=summary,
        assets=results,
        excluded=excluded,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
