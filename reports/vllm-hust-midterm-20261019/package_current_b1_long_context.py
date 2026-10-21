"""Preserve complete current-B1 long-input evidence in bounded Git parts."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
RUN = "m01-recovery-model-long-context-r2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    raw = args.raw.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest["sha256"]
    assert len(raw) == manifest["bytes"]
    expected = {m["name"]: m for m in manifest["members"]}
    assert len(expected) == len(manifest["members"])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        assert set(archive.getnames()) == set(expected)
        for name, member in expected.items():
            content = archive.extractfile(name).read()
            assert hashlib.sha256(content).hexdigest() == member["sha256"]
            assert len(content) == member["bytes"]
        receipt = json.load(archive.extractfile(f"{RUN}/receipt.json"))
        result = json.load(archive.extractfile(f"{RUN}/long-context.json"))
        audit = json.load(archive.extractfile("long-context-audit-r2.json"))
        supplement = json.load(
            archive.extractfile("long-context-final-answer-supplement-r1.json")
        )
    assert receipt["status"] == "measured" and receipt["all_devices_released"]
    assert receipt["bench_exit_code"] == 0
    assert result["completed"] == result["total"] == 6 and result["failed"] == 0
    assert audit["correct"] == result["correct"]
    destination = ROOT / "long-context-current-b1-20261021"
    destination.mkdir(exist_ok=False)
    parts = []
    for index, offset in enumerate(range(0, len(raw), 900000)):
        chunk = raw[offset : offset + 900000]
        name = f"long-context-evidence.tar.gz.part{index:03d}"
        (destination / name).write_bytes(chunk)
        parts.append(
            {
                "name": name,
                "bytes": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            }
        )
    record = {
        "schema": "midterm-current-b1-long-context/v1",
        "archive_sha256": manifest["sha256"],
        "archive_bytes": len(raw),
        "parts": parts,
        "members": manifest["members"],
        "raw_url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
        "m01-recovery-20261021-r7/" + args.raw.name,
        "run_id": RUN,
        "scope": "Current B1 synthetic retrieval ability; not primary-dataset accuracy or throughput gain.",
        "result": {k: result[k] for k in ("completed", "failed", "correct", "total")},
        "audit": audit,
        "supplementary_final_answer_diagnosis": supplement,
    }
    (destination / "archive.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({"parts": len(parts), "bytes": len(raw), "result": record["result"]})
    )


if __name__ == "__main__":
    main()
