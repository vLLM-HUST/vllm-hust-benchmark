"""Package selected original receipts and pin the complete raw release archive."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
RAW = "m01-fenced-full-model-r1-20261022.tar.gz"
RAW_SHA = "f4e8281f3df743e2b907bbb79fbf578fe92ea426f9d3b2dcd9d4d72dc245e214"  # pragma: allowlist secret
AUDIT = "m01-fenced-full-model-r1-audit-20261022.tar.gz"
AUDIT_SHA = "d886bdc523870e9b49d7b2517d336deadc71fe1016197019f748022a38b52a49"  # pragma: allowlist secret
BASE = "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r7/"


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    assert sha(args.source / RAW) == RAW_SHA
    assert sha(args.source / AUDIT) == AUDIT_SHA
    out = ROOT / "m01-recovery/fenced-model-20261022"
    out.mkdir(exist_ok=False)
    members = []
    with tarfile.open(args.source / RAW) as archive:
        for name in (
            "receipt.json",
            "device-envelope.json",
            "compiler-cache-initialization.json",
            "launch-command.json",
            "npu-before.txt",
            "npu-after.txt",
        ):
            member = "m01-recovery-model-fenced-r1/" + name
            raw = archive.extractfile(member).read()
            (out / name).write_bytes(raw)
            members.append(
                dict(
                    name=name,
                    original_member=member,
                    sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
    shutil.copyfile(args.source / AUDIT, out / AUDIT)
    with tarfile.open(args.source / AUDIT) as archive:
        audit = json.load(archive.extractfile("m01-fenced-r1-audit.json"))
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    manifest = dict(
        schema="midterm-fenced-model/v1",
        selected_members=members,
        archives=[
            dict(
                name=name,
                sha256=digest,
                bytes=(args.source / name).stat().st_size,
                raw_url=BASE + name,
                in_git=name == AUDIT,
            )
            for name, digest in ((RAW, RAW_SHA), (AUDIT, AUDIT_SHA))
        ],
        full_arithmetic_coverage_proven=False,
        model_effective_flops=None,
        m01_attainment=None,
    )
    (out / "archive.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
