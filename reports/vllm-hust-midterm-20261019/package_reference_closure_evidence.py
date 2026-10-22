"""Import selected immutable M01 reference audits; keep large raw archives in Release."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
BASE = "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/m01-recovery-20261021-r7/"
ARCHIVES = {
    "m01-worker-comm-flit-bound": "42656f499ddb13649517d1e418421aa33b903ef82d77fb8baf670ffb90772d19",  # pragma: allowlist secret
    "m01-worker-recurrence-reference": "34c34a819689dd2e19c314d143f7507652f2eee3b72329be12867bcb7916135b",  # pragma: allowlist secret
    "m01-worker-wy-reference": "46074283f8e9c04dfa08dee3250eb8dea3ae520bf6f10be1add00f8f598af00e",  # pragma: allowlist secret
    "m01-worker-conv-reference": "1984b963c6d4b0826c7ba73db661e03b928154ddcaaea21955ac34e6d7c77ecb",  # pragma: allowlist secret
    "m01-worker-hbm-reference": "ec3c93e5a75190b6b30d8101504f22b31b9cac01c8f7080153e1924e6b2e1702",  # pragma: allowlist secret
    "m01-worker-resource-terms": "0e57723036698d8b3e5885a32b8c5c018c6f15bd07b79d8545bc0c2f65fb6de0",  # pragma: allowlist secret
}
SELECTED = {
    "m01-worker-comm-flit-bound": {
        "m01-worker-comm-flit-bound.json": "communication.json",
    },
    "m01-worker-recurrence-reference": {
        "workspace/m01-recurrence-reference-worker-r1/audit.json": "recurrence.json",
    },
    "m01-worker-wy-reference": {
        "workspace/m01-wy-reference-worker-r2/audit.json": "wy.json",
    },
    "m01-worker-conv-reference": {
        "workspace/m01-conv-reference-worker-r1/audit.json": "convolution.json",
    },
    "m01-worker-hbm-reference": {
        "workspace/m01-hbm-reference-worker-r1/audit.json": "hbm.json",
        "workspace/m01-hbm-reference-worker-r1/mix-review.json": "hbm-mix-review.json",
    },
    "m01-worker-resource-terms": {
        "workspace/m01-worker-preprocess-map-audit.json": "preprocess-maps.json",
        "workspace/m01-worker-resource-terms.json": "resource-terms.json",
    },
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def package(source, output):
    archives, selected = [], []
    for stem, expected in ARCHIVES.items():
        name = stem + "-20261022.tar.gz"
        path = source / name
        if sha(path) != expected:
            raise ValueError(f"Archive digest mismatch: {name}")
        archives.append(
            dict(
                name=name,
                sha256=expected,
                bytes=path.stat().st_size,
                raw_url=BASE + name,
                in_git=False,
            )
        )
        with tarfile.open(path) as archive:
            for member, destination in SELECTED[stem].items():
                with archive.extractfile(member) as stream:
                    raw = stream.read()
                json.loads(raw)
                selected.append((destination, name, member, raw))
    output.mkdir(parents=True, exist_ok=False)
    members = []
    for name, archive, original, raw in selected:
        (output / name).write_bytes(raw)
        members.append(
            dict(
                name=name,
                archive=archive,
                original_member=original,
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
    manifest = dict(
        schema="midterm-reference-closure-evidence/v1",
        archives=archives,
        selected_members=members,
        formal_b1_admission=False,
        model_effective_flops=None,
        m01_attainment=None,
        status="reference_measurements_completed_final_accounting_review_pending",
    )
    (output / "archive.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "m01-recovery/reference-closure-20261022"
    )
    args = parser.parse_args()
    package(args.source, args.output)
