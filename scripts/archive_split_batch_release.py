"""Freeze the reviewed public Split-Batch sdist result members without extraction."""

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from migrate_devhub_evidence import SECRET
from migrate_mod_git_evidence import invalid_json_paths, verify

REPOSITORY = "vLLM-HUST/vllm-ascend-split-batch-hust"
TAG = "v0.1.3"
ASSET_ID = 592352879
DIGEST = "d3f3a55e69c84a25d62eb29ddd98ae4e616c54f4bd622a011e4641cad007ab90"  # pragma: allowlist secret
PREFIX = "vllm_ascend_split_batch-0.1.3/bench/results/"


def members(raw):
    payloads = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        seen = set()
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) != member.name.rstrip("/")
                or member.name in seen
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unsafe or duplicate archive member")
            seen.add(member.name)
            if member.isfile() and member.name.startswith(PREFIX):
                raw_member = archive.extractfile(member).read()
                if SECRET.search(raw_member):
                    raise ValueError("Credential review required; value suppressed")
                payloads[member.name] = raw_member
    if not payloads:
        raise ValueError("Empty result scope")
    return payloads


def prepare(asset, destination):
    if destination.exists():
        raise ValueError("Refusing to overwrite custody snapshot")
    metadata = json.loads(subprocess.check_output(["gh", "api", f"repos/{REPOSITORY}"]))
    if metadata.get("visibility") != "public" or metadata.get("private"):
        raise ValueError("Public source required")
    release = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{REPOSITORY}/releases/tags/{TAG}"]
        )
    )
    if release["draft"] or release["prerelease"]:
        raise ValueError("Release visibility changed")
    source = next(a for a in release["assets"] if a["id"] == ASSET_ID)
    raw = asset.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest() != DIGEST
        or source["digest"] != "sha256:" + DIGEST
        or len(raw) != source["size"]
    ):
        raise ValueError("Release asset identity mismatch")
    payloads = members(raw)
    if len(payloads) != 75 or sum(map(len, payloads.values())) != 1554773:
        raise ValueError("Reviewed member inventory changed")
    rows = []
    for path, data in sorted(payloads.items()):
        target = destination / "snapshot" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(
            {
                "path": path,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    receipt = {
        "schema": "mod-release-members-custody/v1",
        "source_repository": REPOSITORY,
        "source_visibility": "public",
        "source_release": {"tag": TAG, "draft": False, "prerelease": False},
        "source_asset": {
            k: source[k]
            for k in ("id", "name", "size", "digest", "browser_download_url")
        },
        "scope": {"member_prefix": PREFIX},
        "admission": "custody-only; original failures and cohort limits retained; no new performance qualification",
        "source_retention": "Published installation artifact retained; no overwrite or deletion authorized by member custody alone.",
        "files": rows,
        "invalid_json_paths": invalid_json_paths(payloads),
    }
    (destination / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    verify(destination)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"Verified {len(prepare(args.asset, args.destination)['files'])} original members"
    )
