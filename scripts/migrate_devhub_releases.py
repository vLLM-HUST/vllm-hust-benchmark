"""Copy existing public dev-hub releases to benchmark with digest verification."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

SOURCE = "vLLM-HUST/vllm-hust-dev-hub"
TARGET = "vLLM-HUST/vllm-hust-benchmark"


def gh(*args):
    result = subprocess.run(
        ["gh", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )
    if result.returncode:
        # GitHub download errors can contain signed URLs; never print stderr.
        raise RuntimeError(
            f"GitHub {args[0]} operation failed (exit {result.returncode}); "
            "remote diagnostic URLs suppressed"
        )
    return result.stdout


def releases(repo):
    pages = json.loads(gh("api", "--paginate", "--slurp", f"repos/{repo}/releases"))
    return [row for page in pages for row in page]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def migrate(cache, output, publish):
    source = releases(SOURCE)
    targets = {row["tag_name"]: row for row in releases(TARGET)}
    records = []
    for release in source:
        tag = release["tag_name"]
        if not tag.startswith("szyn-swebench-qwen35-"):
            raise ValueError("Unexpected release outside approved migration scope")
        directory = cache / tag
        directory.mkdir(parents=True, exist_ok=True)
        expected = release["assets"]
        for asset in expected:
            name = asset["name"]
            if Path(name).name != name:
                raise ValueError("Unsafe asset name")
            path = directory / name
            valid = (
                path.is_file()
                and path.stat().st_size == asset["size"]
                and digest(path) == asset["digest"]
            )
            if not valid:
                for attempt in range(3):
                    try:
                        gh(
                            "release",
                            "download",
                            tag,
                            "--repo",
                            SOURCE,
                            "--pattern",
                            name,
                            "--dir",
                            str(directory),
                            "--clobber",
                        )
                        break
                    except RuntimeError:
                        if attempt == 2:
                            raise
                        time.sleep(2**attempt)
            if path.stat().st_size != asset["size"] or digest(path) != asset["digest"]:
                raise ValueError(f"Source asset mismatch: {tag}/{name}")
        if publish:
            if tag not in targets:
                notes = (
                    "Evidence custody migrated from " + release["html_url"] + ".\n\n"
                    "Original status and content are preserved; this is not new evidence "
                    "admission. Source assets are retained until consumers migrate.\n\n"
                    "## Original release notes\n\n" + (release["body"] or "")
                )
                gh(
                    "release",
                    "create",
                    tag,
                    "--repo",
                    TARGET,
                    "--target",
                    "main",
                    "--title",
                    release["name"] or tag,
                    "--notes",
                    notes,
                    "--latest=false",
                )
                targets[tag] = {"assets": []}
            existing = {a["name"]: a for a in targets[tag]["assets"]}
            for asset in expected:
                name = asset["name"]
                if name in existing:
                    if existing[name]["digest"] != asset["digest"]:
                        raise ValueError(f"Destination asset conflict: {tag}/{name}")
                    continue
                gh("release", "upload", tag, str(directory / name), "--repo", TARGET)
            remote = json.loads(gh("api", f"repos/{TARGET}/releases/tags/{tag}"))
            actual = {a["name"]: a for a in remote["assets"]}
            for asset in expected:
                copied = actual[asset["name"]]
                if (
                    copied["digest"] != asset["digest"]
                    or copied["size"] != asset["size"]
                ):
                    raise ValueError("Remote destination digest mismatch")
        records.append(
            {
                "source_release": release["html_url"],
                "source_release_id": release["id"],
                "original_published_at": release.get("published_at"),
                "original_updated_at": release.get("updated_at"),
                "original_target_commitish": release.get("target_commitish"),
                "original_title": release["name"],
                "original_body": release["body"],
                "destination_release": f"https://github.com/{TARGET}/releases/tag/{tag}",
                "remote_verified": publish,
                "assets": [
                    {
                        "name": a["name"],
                        "bytes": a["size"],
                        "digest": a["digest"],
                        "source_url": a["browser_download_url"],
                        "destination_url": f"https://github.com/{TARGET}/releases/download/{tag}/{a['name']}",
                    }
                    for a in expected
                ],
            }
        )
        print(f"{tag}: {len(expected)} assets verified", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(
            {
                "schema": "release-custody-transfer/v1",
                "releases": records,
                "source_cleanup": "pending consumer migration; no source assets deleted",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    if output.exists() and output.read_text() != serialized:
        raise ValueError(
            "Custody snapshot changed; choose a new output instead of rewriting history"
        )
    if not output.exists():
        output.write_text(serialized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    migrate(args.cache, args.output, args.publish)
