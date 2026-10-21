"""Preserve checksum-pinned M01 recovery and collector qualifications in Git."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
ARCHIVES = {
    "m01-r7-map-reconciliation-20261021.tar.gz": "1f429404b0a47a080230d8f156200a1dc91c6a568e32c79b853287305b4e62ef",  # pragma: allowlist secret
    "m01-r7-trace-capability-20261021.tar.gz": "4fb900b431f9d00db91d50889da7100c57df7cc031e1f682a24135349e96c2b2",  # pragma: allowlist secret
    "m01-trace-qualification-r1-evidence.tar.gz": "db117a7a1cc1404307d685aee7a1c096b59e4a003c817fb04e7db3278400d9fc",  # pragma: allowlist secret
    "m01-event-identity-r2-evidence.tar.gz": "b656a1b55e31688363ded2ce8aa544608bb8f5a00bacfdb1d8f3973cce02d3f6",  # pragma: allowlist secret
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    destination = ROOT / "m01-recovery/trace-qualification-20261021"
    destination.mkdir(exist_ok=False)
    records = []
    for name, expected in ARCHIVES.items():
        raw = (args.source / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        members = []
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            seen = set()
            for member in archive:
                path = Path(member.name)
                assert not path.is_absolute() and ".." not in path.parts
                assert member.name not in seen
                seen.add(member.name)
                if member.isdir():
                    continue
                assert member.isfile()
                content = archive.extractfile(member).read()
                members.append(
                    dict(
                        name=member.name,
                        bytes=len(content),
                        sha256=hashlib.sha256(content).hexdigest(),
                    )
                )
        parts = []
        for index, offset in enumerate(range(0, len(raw), 900000)):
            chunk = raw[offset : offset + 900000]
            part_name = f"{name}.part{index:03d}"
            (destination / part_name).write_bytes(chunk)
            parts.append(
                dict(
                    name=part_name,
                    bytes=len(chunk),
                    sha256=hashlib.sha256(chunk).hexdigest(),
                )
            )
        records.append(
            dict(
                name=name,
                bytes=len(raw),
                sha256=expected,
                parts=parts,
                members=members,
                raw_url="https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r7/"
                + name,
            )
        )
    (destination / "archive.json").write_text(
        json.dumps(
            dict(
                schema="midterm-m01-trace-qualification/v1",
                archives=records,
                whole_model_dependency_closure_proven=False,
                model_effective_flops=None,
                m01_attainment=None,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
