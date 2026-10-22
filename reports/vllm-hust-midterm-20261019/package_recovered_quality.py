"""Retain a complete recovered quality run as bounded Git archive parts."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
RUN = "m01-recovery-model-quality-r1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    run = args.run
    manifest = json.loads(args.manifest.read_text())
    raw = args.raw.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["sha256"]
    assert len(raw) == manifest["bytes"]
    members = {m["name"]: m for m in manifest["members"]}
    assert len(members) == len(manifest["members"])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        assert set(archive.getnames()) == set(members)
        for name, expected in members.items():
            content = archive.extractfile(name).read()
            assert hashlib.sha256(content).hexdigest() == expected["sha256"]
            assert len(content) == expected["bytes"]
        receipt = json.load(archive.extractfile(f"{run}/receipt.json"))
        result = json.load(archive.extractfile(f"{run}/mmlu-pro.json"))
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert receipt["bench_exit_code"] == 0
    assert result["completed"] == result["total"] == 200 and result["failed"] == 0
    destination = args.destination or ROOT / "quality-regression/recovered-20261021"
    destination.mkdir(exist_ok=False)
    parts = []
    for index, offset in enumerate(range(0, len(raw), 900000)):
        chunk = raw[offset : offset + 900000]
        name = f"quality-evidence.tar.gz.part{index:03d}"
        (destination / name).write_bytes(chunk)
        parts.append(
            {
                "name": name,
                "bytes": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            }
        )
    record = {
        "schema": "m01-recovered-quality/v1",
        "archive_sha256": manifest["sha256"],
        "archive_bytes": len(raw),
        "parts": parts,
        "members": manifest["members"],
        "raw_url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
        "m01-recovery-20261021-r7/" + args.raw.name,
        "run_id": run,
        "scope": "Same 200 MMLU-Pro questions and frozen scorer in recovered runtime; not a new throughput result or statistical noninferiority claim.",
        "result": {
            k: result[k]
            for k in (
                "completed",
                "failed",
                "correct",
                "total",
                "accuracy",
                "truncated",
                "unparsed",
            )
        },
        "m01_attainment": None,
    }
    (destination / "archive.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({"parts": len(parts), "bytes": len(raw), "result": record["result"]})
    )


if __name__ == "__main__":
    main()
