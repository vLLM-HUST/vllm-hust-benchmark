"""Preserve a terminal M05 parser confirmation in checksum-bound Git parts."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent


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
        run = manifest["run_id"]
        receipt = json.load(archive.extractfile(f"{run}/receipt.json"))
        assert receipt["status"] in ("measured", "failed") and "ended" in receipt
        assert receipt["all_devices_released"]
        result = audit = None
        if receipt["status"] == "measured":
            result = json.load(archive.extractfile(f"{run}/long-context.json"))
            audit = json.load(archive.extractfile("m05-parser-audit-r1.json"))
            assert result["completed"] == result["total"] == 6
            assert result["failed"] == 0 and audit["correct"] == result["correct"]
    destination = ROOT / "m05-parser-confirmation-20261021"
    destination.mkdir(exist_ok=False)
    parts = []
    for index, offset in enumerate(range(0, len(raw), 900000)):
        chunk = raw[offset : offset + 900000]
        name = f"parser-confirmation.tar.gz.part{index:03d}"
        (destination / name).write_bytes(chunk)
        parts.append(
            {
                "name": name,
                "bytes": len(chunk),
                "sha256": hashlib.sha256(chunk).hexdigest(),
            }
        )
    record = {
        "schema": "midterm-m05-parser-confirmation/v1",
        "archive_sha256": manifest["sha256"],
        "archive_bytes": len(raw),
        "parts": parts,
        "members": manifest["members"],
        "run_id": run,
        "raw_url": "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/download/"
        "m01-recovery-20261021-r7/" + args.raw.name,
        "status": receipt["status"],
        "scope": "Synthetic long-input API repair confirmation; not primary-dataset quality or performance",
        "result": {k: result[k] for k in ("completed", "failed", "correct", "total")}
        if result
        else None,
        "audit": audit,
    }
    (destination / "archive.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {"parts": len(parts), "bytes": len(raw), "status": receipt["status"]}
        )
    )


if __name__ == "__main__":
    main()
