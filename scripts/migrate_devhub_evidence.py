"""Prepare byte-preserving dev-hub custody transfer; never delete sources."""

import argparse
import collections
import gzip
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

PREFIXES = (
    "results/",
    "docs/evidence/",
    "Ascend-Machine/results/",
    "Ascend-Machine/910B2/results/",
)
REPOSITORY = "vLLM-HUST/vllm-hust-benchmark"
TAG = "evidence-devhub-20261020-v1"
ASSET = "devhub-evidence-20261020.tar.gz"
SECRET = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    rb"|gh[pousr]_[A-Za-z0-9]{30,}"
    rb"|Authorization:\s*Bearer\s+(?!\$|<)[A-Za-z0-9._-]{20,}",
    re.IGNORECASE,
)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def tree(repo, ref):
    result = {}
    for record in git(repo, "ls-tree", "-rz", ref).split(b"\0"):
        if record:
            metadata, path = record.split(b"\t", 1)
            if metadata.split()[1] == b"blob":
                result[path.decode()] = metadata.split()[2].decode()
    return result


def prepare(source, benchmark, source_ref, output, bundle):
    if output.exists() or bundle.exists():
        raise ValueError("Refusing to overwrite a custody snapshot")
    commit = git(source, "rev-parse", source_ref).decode().strip()
    base = git(benchmark, "rev-parse", "HEAD").decode().strip()
    existing = collections.defaultdict(list)
    for path, oid in tree(benchmark, base).items():
        existing[oid].append(path)
    rows = []
    outside = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, oid in sorted(tree(source, commit).items()):
            if not path.startswith(PREFIXES):
                if any(
                    word in path.lower() for word in ("result", "receipt", "evidence")
                ):
                    outside.append(path)
                continue
            raw = git(source, "cat-file", "blob", oid)
            scan = gzip.decompress(raw) if path.endswith(".gz") else raw
            if SECRET.search(scan):
                raise ValueError(
                    f"Credential review required in {path}; value suppressed"
                )
            row = {"path": path, "bytes": len(raw), "sha256": sha(raw), "git_blob": oid}
            if oid in existing:
                row["destination"] = {
                    "kind": "git",
                    "commit": base,
                    "path": sorted(existing[oid])[0],
                }
            else:
                member = tarfile.TarInfo(path)
                member.size = len(raw)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(raw))
                row["destination"] = {
                    "kind": "release-member",
                    "tag": TAG,
                    "asset": ASSET,
                    "member": path,
                }
            rows.append(row)
    packed = gzip.compress(buffer.getvalue(), mtime=0)
    manifest = {
        "schema": "cross-repository-evidence-custody/v1",
        "source_repository": "vLLM-HUST/vllm-hust-dev-hub",
        "source_commit": commit,
        "destination_repository": REPOSITORY,
        "scope": list(PREFIXES),
        "asset": {
            "url": f"https://github.com/{REPOSITORY}/releases/download/{TAG}/{ASSET}",
            "sha256": sha(packed),
            "bytes": len(packed),
        },
        "files": rows,
        "outside_scope_review": outside,
        "admission": "custody-only; no new measurement, pairing or performance claim",
    }
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_bytes(packed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    verify(manifest, packed, benchmark)
    print(
        json.dumps(
            {
                "files": len(rows),
                "destinations": dict(
                    collections.Counter(row["destination"]["kind"] for row in rows)
                ),
                "archive_bytes": len(packed),
                "sha256": sha(packed),
            }
        )
    )


def verify(manifest, packed, benchmark):
    if sha(packed) != manifest["asset"]["sha256"]:
        raise ValueError("Archive SHA256 mismatch")
    if len(packed) != manifest["asset"]["bytes"]:
        raise ValueError("Archive length mismatch")
    with tarfile.open(fileobj=io.BytesIO(packed), mode="r:gz") as archive:
        members = archive.getmembers()
        expected = [
            r for r in manifest["files"] if r["destination"]["kind"] == "release-member"
        ]
        if len(members) != len(expected) or {m.name for m in members} != {
            r["destination"]["member"] for r in expected
        }:
            raise ValueError("Archive member set mismatch")
        for row in manifest["files"]:
            destination = row["destination"]
            if destination["kind"] == "git":
                raw = git(
                    benchmark, "show", f"{destination['commit']}:{destination['path']}"
                )
            else:
                member = archive.getmember(destination["member"])
                if not member.isfile():
                    raise ValueError("Not a regular file")
                raw = archive.extractfile(member).read()
            if sha(raw) != row["sha256"] or len(raw) != row["bytes"]:
                raise ValueError(f"Evidence mismatch: {row['path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-ref", default="origin/main")
    parser.add_argument(
        "--benchmark", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.benchmark, args.source_ref, args.output, args.bundle)
