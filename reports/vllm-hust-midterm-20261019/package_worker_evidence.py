"""Verify immutable worker evidence archives and package selected original receipts."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
BASE = "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r7/"
ARCHIVES = {
    "m01-worker-full-model-r1": "afba33d24d1767a1c7f295b5a99eb54a7dfc7e5b5d8df20a0f3e673cc7189b37",  # pragma: allowlist secret
    "m01-worker-full-model-r1-audit": "6fd7432b3f3801fc83a78d7bdf28977c7257c906f1845a553771681a3ae760b0",  # pragma: allowlist secret
    "m01-worker-r1-accounting": "ddb0a97eaa43a624b83f1f2615b46a0f2410430fe78ab98e0fedac876473149d",  # pragma: allowlist secret
    "m01-worker-moe-reference": "5938b6394a043fcf1fb330dd150b8475b44c152ca03bc8be50c37b0e44ef8ec9",  # pragma: allowlist secret
    "m01-worker-linear-reference": "d0d855665d8cbeb7fe9da8c6b33047ad18bae4f17d0bc98ec33d8725eb62a583",  # pragma: allowlist secret
    "m01-worker-fia-reference": "c346a59c67076adf22a3bb422338a188d241c8c456b38eecba89a8097038825b",  # pragma: allowlist secret
    "m01-worker-gdn-reference": "8c360a3ce43deb23a020cbc36e81a7c8717da5d6e1a601cadc57d8f0a3d862fb",  # pragma: allowlist secret
    "m01-worker-aux-reference": "8fe5c98f8a6d069cb9000ad8b2f7b3c76b77ee3456f6451d8eca12c88ab03eb5",  # pragma: allowlist secret
    "m01-worker-activation-collective-audit": "85d3dcc39ab83be69f310f2025522fb464db8e118cfb25a0f6d9635bff8cbcc8",  # pragma: allowlist secret
    "m01-worker-collective-reference": "e482d8b1067ae33bf1256c3d14aeb99d94ce6279e55059c72312e1b79caa2da3",  # pragma: allowlist secret
    "m01-worker-hccs-unit-audit": "3ce63cb94bb35e44518162b8c170057aaa76125b61d8bf086ee8ebdaf962f99e",  # pragma: allowlist secret
}
SELECTED = {
    "m01-worker-r1-accounting": ["recovery_model_worker_r1-accounted-families.json"],
    "m01-worker-full-model-r1-audit": ["m01-worker-r1-audit.json"],
    "m01-worker-collective-reference": [
        "workspace/m01-worker-collective-reference-joined.json",
    ],
    "m01-worker-hccs-unit-audit": [
        "m01-worker-hccs-unit-audit.json",
        "m01-worker-hccs-phase-unit-audit.json",
    ],
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package(source, output):
    archives = []
    # Validate every source before creating the output directory.
    for stem, digest in ARCHIVES.items():
        name = stem + "-20261022.tar.gz"
        path = source / name
        if sha(path) != digest:
            raise ValueError(f"Archive digest mismatch: {name}")
        archives.append(
            dict(
                name=name,
                sha256=digest,
                bytes=path.stat().st_size,
                raw_url=BASE + name,
                in_git=False,
            )
        )
    output.mkdir(exist_ok=False, parents=True)
    members = []
    for stem, names in SELECTED.items():
        archive_name = stem + "-20261022.tar.gz"
        with tarfile.open(source / archive_name) as archive:
            for member in names:
                with archive.extractfile(member) as stream:
                    raw = stream.read()
                name = Path(member).name
                (output / name).write_bytes(raw)
                members.append(
                    dict(
                        name=name,
                        archive=archive_name,
                        original_member=member,
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
    manifest = dict(
        schema="midterm-worker-evidence/v1",
        archives=archives,
        selected_members=members,
        formal_b1_admission=False,
        full_arithmetic_coverage_proven=False,
        model_effective_flops=None,
        m01_attainment=None,
    )
    (output / "archive.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "m01-recovery/worker-model-20261022"
    )
    args = parser.parse_args()
    package(args.source, args.output)
