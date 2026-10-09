"""Retrieve pinned public Terminal-Bench source and image metadata, not layers.

Anonymous Docker Hub quotas apply. Completed files are hash-checked and reused.
No task code is executed. Image configuration files stay local and unpublished.
"""

from __future__ import annotations

import argparse
import gzip
import json
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from audit_pyramidkv_terminal import (
    REVISION,
    TREE_SHA256,
    audit_image,
    checked,
    sha,
    verify_snapshot,
)

DEFAULT_REPORT = (
    Path(__file__).parents[1] / "reports/pyramidkv-terminal-source-audit-20261009"
)
ACCEPT = (
    "application/vnd.oci.image.index.v1+json,"
    "application/vnd.docker.distribution.manifest.list.v2+json,"
    "application/vnd.oci.image.manifest.v1+json,"
    "application/vnd.docker.distribution.manifest.v2+json"
)


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (
            redirected is not None
            and urllib.parse.urlsplit(req.full_url).netloc
            != urllib.parse.urlsplit(newurl).netloc
        ):
            redirected.remove_header("Authorization")
        return redirected


def get(url: str, headers: dict | None = None) -> bytes:
    opener = urllib.request.build_opener(PublicRedirect())
    with opener.open(
        urllib.request.Request(url, headers=headers or {}), timeout=90
    ) as response:
        return response.read()


def save_checked(
    path: Path, digest: str, url: str, headers: dict | None = None
) -> bytes:
    if path.exists():
        return checked(path, digest)
    data = get(url, headers)
    if sha(data) != digest.removeprefix("sha256:"):
        raise ValueError("Downloaded content does not match frozen digest")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def extract_snapshot(archive: Path, destination: Path) -> None:
    # Extract only ordinary files; reject links, traversal and duplicate paths.
    seen = set()
    with tarfile.open(archive) as tar:
        for entry in tar:
            path = PurePosixPath(entry.name)
            parts = path.parts[1:]
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("Unsafe archive path")
            if entry.isdir():
                continue
            if not entry.isfile() or not parts or tuple(parts) in seen:
                raise ValueError("Unsupported or duplicate archive entry")
            seen.add(tuple(parts))
            target = destination.joinpath(*parts)
            if target.is_symlink() or any(p.is_symlink() for p in target.parents):
                raise ValueError("Symlink in destination")
            data = tar.extractfile(entry).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Refusing to replace changed snapshot file")
            target.write_bytes(data)


def fetch(
    assets: Path,
    report: Path,
    *,
    source_only: bool = False,
    tasks: list[str] | None = None,
) -> dict:
    tree_bytes = gzip.decompress((report / "tree.json.gz").read_bytes())
    if sha(tree_bytes) != TREE_SHA256:
        raise ValueError("Published source tree changed")
    retrieval = json.loads((report / "retrieval.json").read_text())
    if (
        retrieval["revision"] != REVISION
        or retrieval["repo"] != "harbor-framework/terminal-bench-2-1"
    ):
        raise ValueError("Source retrieval identity mismatch")
    assets.mkdir(parents=True, exist_ok=True)
    tree_path = assets / "tree.json"
    if tree_path.exists():
        checked(tree_path, TREE_SHA256)
    else:
        tree_path.write_bytes(tree_bytes)
    archive = assets / "snapshot.tar.gz"
    save_checked(
        archive,
        retrieval["archive_sha256"],
        "https://codeload.github.com/harbor-framework/terminal-bench-2-1/tar.gz/"
        + REVISION,
    )
    extract_snapshot(archive, assets / "snapshot")
    verify_snapshot(assets / "snapshot", json.loads(tree_bytes))
    if source_only:
        return {
            "source_verified": True,
            "images_verified": 0,
            "image_layers_pulled": False,
            "tasks_executed": False,
        }
    receipts = json.loads(
        gzip.decompress((report / "image-receipts.json.gz").read_bytes())
    )
    if tasks is not None:
        if set(tasks) - {r["task_id"] for r in receipts}:
            raise ValueError("Unknown task identifier")
        receipts = [r for r in receipts if r["task_id"] in tasks]
    for receipt in receipts:
        folder = assets / "image-audit" / receipt["task_id"]
        repository = receipt["image"].split(":", 1)[0]
        if not repository.startswith("alexgshaw/"):
            raise ValueError("Unexpected public registry repository")
        file_specs = [
            ("tag-manifest.json", receipt["tag_digest"], "manifests/"),
            ("platform-manifest.json", receipt["manifest_digest"], "manifests/"),
            ("image-config.json", receipt["config_digest"], "blobs/"),
        ]
        headers = {}
        if any(not (folder / name).exists() for name, _, _ in file_specs):
            query = urllib.parse.urlencode(
                {
                    "service": "registry.docker.io",
                    "scope": "repository:" + repository + ":pull",
                }
            )
            token = json.loads(get("https://auth.docker.io/token?" + query))["token"]
            headers = {"Authorization": "Bearer " + token, "Accept": ACCEPT}
        for name, digest, kind in file_specs:
            if (
                name == "platform-manifest.json"
                and digest == receipt["tag_digest"]
                and not (folder / name).exists()
            ):
                (folder / name).write_bytes(
                    checked(folder / "tag-manifest.json", digest)
                )
            else:
                save_checked(
                    folder / name,
                    digest,
                    "https://registry-1.docker.io/v2/"
                    + repository
                    + "/"
                    + kind
                    + digest,
                    headers,
                )
        audit_image(folder, receipt)
        # Task IDs can contain dots: with_suffix would truncate their names.
        receipt_path = folder.parent / (receipt["task_id"] + ".json")
        serialized = json.dumps(receipt, indent=2) + "\n"
        if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
            raise ValueError("Refusing to replace changed image receipt")
        if not receipt_path.exists():
            receipt_path.write_text(serialized)
        print("Verified image metadata: " + receipt["task_id"], flush=True)
    return {
        "source_verified": True,
        "images_verified": len(receipts),
        "image_layers_pulled": False,
        "tasks_executed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--task", action="append", dest="tasks")
    args = parser.parse_args()
    print(
        json.dumps(
            fetch(
                args.assets, args.report, source_only=args.source_only, tasks=args.tasks
            ),
            indent=2,
        )
    )
