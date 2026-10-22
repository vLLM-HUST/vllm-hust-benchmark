"""Import the immutable captured-window assessment without modifying raw evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
NAME = "m01-worker-accounting-assessment-20261022.tar.gz"
SHA256 = "aa515aa2a6982ba45cd0de896fe127c710f2ca8ae6d4c87babcfb894171d0e38"  # pragma: allowlist secret
BASE = "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r7/"
MEMBERS = {
    "workspace/m01-worker-final-accounting.json": "assessment.json",
    "workspace/m01-worker-composition-review.json": "composition.json",
    "workspace/m01-worker-window-review.json": "window.json",
}


def package(source, output):
    with source.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != SHA256:
        raise ValueError("Assessment archive digest mismatch")
    selected = []
    with tarfile.open(source) as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        for member, name in MEMBERS.items():
            raw = archive.extractfile(member).read()
            checksum = hashlib.sha256(raw).hexdigest()
            assert checksum == manifest["files"][member]
            json.loads(raw)
            selected.append((name, member, raw, checksum))
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for name, member, raw, checksum in selected:
        (output / name).write_bytes(raw)
        entries.append(dict(name=name, original_member=member, sha256=checksum))
    receipt = dict(
        schema="midterm-captured-window-assessment/v1",
        archive=dict(
            name=NAME,
            sha256=SHA256,
            bytes=source.stat().st_size,
            raw_url=BASE + NAME,
            in_git=False,
        ),
        selected_members=entries,
        formal_b1_admission=False,
        uninstrumented_request_only_attainment=None,
    )
    (output / "archive.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "m01-recovery/accounting-assessment-20261022",
    )
    args = parser.parse_args()
    package(args.source, args.output)
