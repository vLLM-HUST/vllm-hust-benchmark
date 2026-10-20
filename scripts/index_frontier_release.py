"""Index existing Frontier release bytes without republishing the archives."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

TAG = "evidence-frontier-20260927-v1"
ASSETS = (
    (
        "phase9-unified-r3-results.tar.gz",
        "benchmark-archive-native-20261020.tar.gz",
        "bb5194fc2090f914d550f4f76b6b1ba76563baaf802870e3640dd52f8118971d",  # pragma: allowlist secret
    ),
    (
        "phase9-unified-retry-r1-results.tar.gz",
        "benchmark-archive-candidates-20261020.tar.gz",
        "2655ee0138afc296962bc3922e4e3f262736dfeb164b023f6af576fa7455c5c9",  # pragma: allowlist secret
    ),
    (
        "phase10-followup-r1-results.tar.gz",
        "benchmark-archive-followup-20261020.tar.gz",
        "49092d97fdfb544ad15fa5ca5b154f9ecbb52b423ec8212cf15d0d7995ffaaf6",  # pragma: allowlist secret
    ),
)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def index_archive(raw, expected):
    if sha(raw) != expected:
        raise ValueError("Archive does not match the uploaded release digest")
    files, runs, seen = [], [], set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe archive member")
            if str(path) in seen:
                raise ValueError("Duplicate archive member")
            seen.add(str(path))
            content = archive.extractfile(member).read()
            files.append(
                {"member": member.name, "bytes": len(content), "sha256": sha(content)}
            )
            if path.name == "status.json":
                status = json.loads(content)
                runs.append(
                    {
                        "member": member.name,
                        "sha256": sha(content),
                        "original_status": status,
                    }
                )
    return files, runs


def generate(source, destination):
    if destination.exists():
        raise ValueError("Refusing to overwrite a release index")
    assets = []
    for original, asset, digest in ASSETS:
        raw = (source / original).read_bytes()
        files, runs = index_archive(raw, digest)
        assets.append(
            {
                "original_name": original,
                "asset_name": asset,
                "url": f"https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/{TAG}/{asset}",
                "bytes": len(raw),
                "sha256": digest,
                "files": files,
                "run_statuses": runs,
            }
        )
    manifest = {
        "schema": "release-evidence-index/v1",
        "release_tag": TAG,
        "assets": assets,
        "admission": "custody-only; original statuses are not new admission decisions",
        "pairing_rule": "Keep treatments, configurations, campaigns and failed attempts distinct.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "archives": len(assets),
                "files": sum(len(a["files"]) for a in assets),
                "run_statuses": sum(len(a["run_statuses"]) for a in assets),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)
