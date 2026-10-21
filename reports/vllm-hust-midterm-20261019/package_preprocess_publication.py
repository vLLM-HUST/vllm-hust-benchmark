"""Build portable qualification evidence from a verified raw release archive."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    with args.archive.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == manifest["sha256"]
    destination = (
        Path(__file__).resolve().parent / "m01-recovery/preprocess-publication-20261021"
    )
    destination.mkdir(exist_ok=False)
    selected = {}
    with tarfile.open(args.archive) as archive:
        for member in archive:
            assert member.isfile() and member.name in manifest["members"]
            data = archive.extractfile(member).read()
            assert (
                hashlib.sha256(data).hexdigest()
                == manifest["members"][member.name]["sha256"]
            )
            if not member.name.endswith(".pt"):
                selected[member.name] = data
    selected["raw-archive-manifest.json"] = args.manifest.read_bytes()
    packed = destination / "evidence.tar.xz"
    with tarfile.open(packed, "w:xz") as archive:
        for name, data in sorted(selected.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(data), 0o644
            archive.addfile(member, io.BytesIO(data))
    summary = {
        "schema": "m01-preprocess-publication/v1",
        "archive": packed.name,
        "archive_sha256": hashlib.sha256(packed.read_bytes()).hexdigest(),
        "members": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(selected.items())
        },
        "raw_archive": {
            "url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
            "m01-recovery-20261021-r7/" + manifest["archive"],
            "sha256": manifest["sha256"],
            "bytes": manifest["bytes"],
        },
        "full_model": False,
        "historical_device_maps_recovered": False,
        "model_effective_flops": None,
        "m01_attainment": None,
    }
    (destination / "archive.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"members": len(selected), "bytes": packed.stat().st_size}))


if __name__ == "__main__":
    main()
