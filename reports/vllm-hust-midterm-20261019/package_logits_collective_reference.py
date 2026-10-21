"""Package the recovered logits calibration without rewriting raw evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
RAW_SHA = "54d38454e4e091ce9a3acce38ec2974313e049d0d36e75dcf530cee28019cd40"  # pragma: allowlist secret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.raw.read_bytes()).hexdigest() == RAW_SHA
    selected, members = {}, {}
    with tarfile.open(args.raw) as archive:
        for member in archive:
            assert member.isfile()
            raw = archive.extractfile(member).read()
            members[member.name] = {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
            if not member.name.endswith(".jsonl"):
                selected[member.name] = raw
    summary = json.loads(selected["recovery-logits-collective-control-r1/summary.json"])
    assert len(summary["shapes"]) == 76 and len(summary["phases"]) == 228
    assert all(s["counter_qualified_repetitions"] == 3 for s in summary["shapes"])
    assert summary["model_effective_flops"] is summary["m01_attainment"] is None
    destination = ROOT / "m01-recovery/logits-collective-reference-20261021"
    destination.mkdir(exist_ok=False)
    packed = destination / "evidence.tar.xz"
    with tarfile.open(packed, "w:xz") as archive:
        for name, raw in sorted(selected.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mode = len(raw), 0o644
            archive.addfile(member, io.BytesIO(raw))
    index = {
        "schema": "m01-logits-collective-reference/v1",
        "archive": packed.name,
        "archive_sha256": hashlib.sha256(packed.read_bytes()).hexdigest(),
        "members": {name: members[name] for name in sorted(selected)},
        "raw_archive": {
            "url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
            "m01-recovery-20261021-r7/" + args.raw.name,
            "sha256": RAW_SHA,
            "bytes": args.raw.stat().st_size,
            "members": members,
        },
        "scope": "Source-shaped logits collectives, not full-model HCCL coverage.",
        "model_effective_flops": None,
        "m01_attainment": None,
    }
    (destination / "archive.json").write_text(json.dumps(index, indent=2) + "\n")
    print(
        json.dumps({"portable_bytes": packed.stat().st_size, "members": len(selected)})
    )


if __name__ == "__main__":
    main()
