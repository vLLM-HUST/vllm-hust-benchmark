"""Explicit, visibility-preserving MOD Release transfer with download-back checks."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from migrate_devhub_releases import (
    check_visibility,
    digest,
    gh,
    releases,
    visibility_flags,
)

TARGET = "vLLM-HUST/vllm-hust-benchmark"
APPROVED = {
    "vLLM-HUST/vllm-hust-kv-materialization-arrival-control": {
        "g0-issue17-stop-20260901-v2",
        "wavemat-m0-raw-20260824-v1",
        "m2-issue3-closure-evidence-20260811",
    },
    "vLLM-HUST/vllm-ascend-MuxPD-hust": {
        "evidence-tp4-groupfx-20261012",
        "evidence-tp4-goodput-20261011",
    },
}


def validate_asset(asset):
    name = asset["name"]
    if Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("Unsafe asset name")
    if not asset.get("digest", "").startswith("sha256:"):
        raise ValueError("Missing source digest")


def remote_digest(repo, asset):
    process = subprocess.Popen(
        [
            "gh",
            "api",
            f"repos/{repo}/releases/assets/{asset['id']}",
            "-H",
            "Accept: application/octet-stream",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    hasher = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        hasher.update(chunk)
        size += len(chunk)
    if process.wait() != 0:
        raise RuntimeError("Download-back failed; diagnostics suppressed")
    if (size, "sha256:" + hasher.hexdigest()) != (asset["size"], asset["digest"]):
        raise ValueError("Download-back size/digest mismatch")


def migrate(source, output):
    if source not in APPROVED:
        raise ValueError("Source outside reviewed scope")
    for repo in (source, TARGET):
        metadata = json.loads(gh("api", f"repos/{repo}"))
        if metadata["visibility"] != "public":
            raise ValueError("This transfer only permits public-to-public custody")
    targets = {r["tag_name"]: r for r in releases(TARGET)}
    sources = {r["tag_name"]: r for r in releases(source)}
    records = []
    for tag in sorted(APPROVED[source]):
        original = sources[tag]
        for asset in original["assets"]:
            validate_asset(asset)
        copied = targets.get(tag)
        if copied is None:
            gh(
                "release",
                "create",
                tag,
                "--repo",
                TARGET,
                "--target",
                "main",
                "--title",
                original["name"] or tag,
                "--latest=false",
                "--notes",
                f"Canonical evidence custody from {original['html_url']}. "
                "Original bytes, status and validity retained; not a new experiment. "
                "Source copies remain until consumer migration and verification.\n\n"
                + (original["body"] or ""),
                *visibility_flags(original),
            )
            copied = json.loads(gh("api", f"repos/{TARGET}/releases/tags/{tag}"))
        check_visibility(original, copied)
        if original["html_url"] not in (copied["body"] or ""):
            raise ValueError("Destination tag collision: no matching source identity")
        existing = {a["name"]: a for a in copied["assets"]}
        verified = []
        for asset in original["assets"]:
            name = asset["name"]
            with tempfile.TemporaryDirectory(prefix="mod-custody-") as directory:
                path = Path(directory) / name
                with path.open("wb") as sink:
                    subprocess.run(
                        [
                            "gh",
                            "api",
                            f"repos/{source}/releases/assets/{asset['id']}",
                            "-H",
                            "Accept: application/octet-stream",
                        ],
                        stdout=sink,
                        stderr=subprocess.DEVNULL,
                        check=True,
                    )
                if (
                    path.stat().st_size != asset["size"]
                    or digest(path) != asset["digest"]
                ):
                    raise ValueError("Source size/digest mismatch")
                if name not in existing:
                    gh("release", "upload", tag, str(path), "--repo", TARGET)
                current = json.loads(
                    gh("api", f"repos/{TARGET}/releases/{copied['id']}")
                )
                check_visibility(original, current)
                target = next(a for a in current["assets"] if a["name"] == name)
                if (target["size"], target["digest"]) != (
                    asset["size"],
                    asset["digest"],
                ):
                    raise ValueError("Target size/digest mismatch; never overwrite")
                remote_digest(TARGET, target)
                verified.append(
                    {
                        "name": name,
                        "bytes": asset["size"],
                        "digest": asset["digest"],
                        "source_id": asset["id"],
                        "source_url": asset["browser_download_url"],
                        "destination_id": target["id"],
                        "destination_url": target["browser_download_url"],
                        "download_back_verified": True,
                    }
                )
                print(tag, name, "verified", flush=True)
        records.append(
            {
                "tag": tag,
                "source": source,
                "source_release_id": original["id"],
                "source_url": original["html_url"],
                "source_target_commitish": original["target_commitish"],
                "original_body": original["body"],
                "draft": original["draft"],
                "prerelease": original["prerelease"],
                "destination_release_id": copied["id"],
                "destination_url": copied["html_url"],
                "assets": verified,
            }
        )
    if output.exists():
        raise ValueError("Use a fresh receipt path; do not overwrite history")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema": "mod-release-custody/v1",
                "releases": records,
                "source_cleanup": "not performed; pending consumer migration",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=APPROVED)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    migrate(args.source, args.output)
