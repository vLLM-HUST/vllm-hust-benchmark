"""Validate the public runtime lock and optionally compare installed versions.

This does not install packages, import neural libraries, load models or claim a
fresh runtime rebuild. Run installed verification using the model environment's
Python, not an unrelated client interpreter.
"""

import argparse
from collections import Counter
import importlib.metadata as metadata
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


def secure_url(value, host):
    url = urlsplit(value)
    if (
        url.scheme != "https"
        or url.netloc != host
        or url.username
        or url.password
        or url.query
        or url.fragment
        or any(c.isspace() for c in value)
    ):
        raise ValueError("credential-free HTTPS publisher URL required")


def sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("SHA256 required")


def load_lock(path):
    lock = json.loads(Path(path).read_text("utf-8"))
    if lock.get("schema_version") != "full-duplex-model-runtime-lock/v1":
        raise ValueError("full duplex model-runtime lock required")
    archives = {}
    for archive in lock["publisher_archives"]:
        name = archive["name"]
        if name in archives or Path(name).name != name:
            raise ValueError("unique archive basename required")
        secure_url(archive["url"], "github.com")
        if not archive["url"].startswith(
            "https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.9.0/"
        ):
            raise ValueError("pinned publisher release required")
        sha(archive["sha256"])
        members = {}
        for member in archive["members"]:
            path = member["path"]
            if (
                path.startswith("/")
                or ".." in path.split("/")
                or "\\" in path
                or path in members
            ):
                raise ValueError("safe unique archive member required")
            sha(member["sha256"])
            members[path] = member
        archives[name] = members
    names = set()
    counts = Counter()
    protected = {
        re.sub(r"[-_.]+", "-", n).lower() for n in lock["protected_core_versions"]
    }
    if protected != {
        "torch",
        "torch-npu",
        "torchaudio",
        "torchvision",
        "numpy",
        "triton-ascend",
    }:
        raise ValueError("six protected core packages required")
    for row in lock["packages"]:
        name = row["name"]
        if (
            name in names
            or name in protected
            or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
        ):
            raise ValueError("duplicate, noncanonical or protected package")
        names.add(name)
        counts[row["binding"]] += 1
        if row["binding"] == "PUBLIC_HASHED_ARTIFACT":
            secure_url(row["url"], "files.pythonhosted.org")
            sha(row["sha256"])
        elif row["binding"] == "PUBLISHER_ARCHIVE_WHEEL":
            member = archives[row["archive"]][row["member"]]
            if (
                member["sha256"] != row["sha256"]
                or Path(row["member"]).name != row["filename"]
            ):
                raise ValueError("wheel binding differs from publisher archive member")
        elif row["binding"] == "EDITABLE_SOURCE":
            if (
                name not in {"sglang", "sglang-omni"}
                or row["source_binding"] not in lock["sources"]
            ):
                raise ValueError("explicit editable source binding required")
        else:
            raise ValueError("unresolved runtime artifact")
    if counts != {
        "PUBLIC_HASHED_ARTIFACT": 177,
        "PUBLISHER_ARCHIVE_WHEEL": 5,
        "EDITABLE_SOURCE": 2,
    }:
        raise ValueError("complete audited 184-package inventory required")
    return lock


def verify_versions(lock):
    expected = {row["name"]: row["version"] for row in lock["packages"]}
    expected.update(lock["protected_core_versions"])
    mismatches = {}
    for name, version in expected.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        if actual != version:
            mismatches[name] = {"expected": version, "actual": actual}
    if mismatches:
        raise ValueError(
            "installed runtime versions differ: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return len(expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/workload-data/validation/full-duplex-model-runtime-lock-20261107.json",
    )
    parser.add_argument("--verify-installed", action="store_true")
    args = parser.parse_args()
    lock = load_lock(args.lock)
    report = {
        "status": "LOCK_VALIDATED",
        "runtime_packages": len(lock["packages"]),
        "publisher_archives": len(lock["publisher_archives"]),
        "neural_execution": False,
        "fresh_runtime_rebuild_verified": False,
        "source_checkbox_eligible": False,
    }
    if args.verify_installed:
        report.update(
            status="INSTALLED_VERSION_METADATA_VERIFIED",
            compared_packages=verify_versions(lock),
        )
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
