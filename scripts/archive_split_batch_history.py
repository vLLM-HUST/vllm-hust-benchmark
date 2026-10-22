"""Freeze reviewed public Split-Batch Release evidence directories verbatim."""

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
ASSETS = {
    "0.1.0": (
        590594140,
        "4d309887cac9b0b99da72bcf13f833504f86d99201c9c499a21ae030738ab9b3",  # pragma: allowlist secret
    ),
    "0.1.1": (
        590589346,
        "7e364f13c59d4fa0e6e36305d6e44821e9eee4080e219b14f9e107120431acd3",  # pragma: allowlist secret
    ),
    "0.1.2": (
        591015498,
        "418c84cc8419f6ea0261295b208fdbb350f06211c07724d7864f14cec735fd6e",  # pragma: allowlist secret
    ),
    "0.1.3": (
        592352879,
        "d3f3a55e69c84a25d62eb29ddd98ae4e616c54f4bd622a011e4641cad007ab90",  # pragma: allowlist secret
    ),
}


def members(raw, root, prefixes):
    payloads, seen = {}, set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or str(path) != member.name.rstrip("/")
                or member.name in seen
                or path.parts[0] != root
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError("Unsafe or duplicate source member")
            seen.add(member.name)
            relative = path.relative_to(root).as_posix()
            if member.isfile() and relative.startswith(prefixes):
                data = archive.extractfile(member).read()
                if SECRET.search(data):
                    raise ValueError("Credential review required; value suppressed")
                payloads[member.name] = data
    if not payloads:
        raise ValueError("Empty evidence scope")
    return payloads


def prepare(asset, version, destination):
    if destination.exists():
        raise ValueError("Refusing to overwrite custody")
    metadata = json.loads(subprocess.check_output(["gh", "api", f"repos/{REPOSITORY}"]))
    if metadata.get("visibility") != "public" or metadata.get("private"):
        raise ValueError("Public source required")
    release = json.loads(
        subprocess.check_output(
            ["gh", "api", f"repos/{REPOSITORY}/releases/tags/v{version}"]
        )
    )
    if release["draft"] or release["prerelease"]:
        raise ValueError("Visibility changed")
    ident, digest = ASSETS[version]
    source = next(a for a in release["assets"] if a["id"] == ident)
    raw = asset.read_bytes()
    if (
        source["digest"] != "sha256:" + digest
        or hashlib.sha256(raw).hexdigest() != digest
        or len(raw) != source["size"]
    ):
        raise ValueError("Asset identity mismatch")
    prefixes = (
        ("docs/evidence/",)
        if version == "0.1.3"
        else ("bench/results/", "docs/evidence/")
    )
    payloads = members(raw, f"vllm_ascend_split_batch-{version}", prefixes)
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
        "source_release": {"tag": "v" + version, "draft": False, "prerelease": False},
        "source_asset": {
            k: source[k]
            for k in ("id", "name", "size", "digest", "browser_download_url")
        },
        "scope": {"member_prefixes": list(prefixes)},
        "admission": "custody only; historical code, failures and cohort limits retained; no new qualification",
        "source_retention": "Installation assets retained pending consumer review; never rewrite published bytes.",
        "files": rows,
        "invalid_json_paths": invalid_json_paths(payloads),
    }
    (destination / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    verify(destination)
    print(
        json.dumps(
            {
                "version": version,
                "files": len(rows),
                "bytes": sum(r["bytes"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--version", choices=ASSETS, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.asset, args.version, args.destination)
