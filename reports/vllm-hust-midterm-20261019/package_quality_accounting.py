"""Package token-extent qualification or paired full-answer quality evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("accounting", "quality"), required=True)
    parser.add_argument("--run", action="append", default=[])
    args = parser.parse_args()
    if args.kind == "accounting":
        directories = ["microflow-token-accounting-r1", "token-observer"]
        sources = [
            "prepare_token_observer.py",
            "runtime-token-accounting.sh",
            "serve-token-accounting.sh",
            "measure_token_accounting.py",
            "launch_token_accounting.py",
            "token-observer-build.json",
            "test_token_observer.py",
            "audit_token_accounting.py",
            "audit_token_accounting_drained.py",
            "token-accounting-summary-r1.json",
            "token-accounting-drained-summary-r1.json",
        ]
        summaries = {
            "request-window-summary.json": "token-accounting-summary-r1.json",
            "drained-summary.json": "token-accounting-drained-summary-r1.json",
        }
    else:
        assert args.run, "Quality packaging requires explicit run directories"
        directories = args.run
        sources = [
            "quality_client.py",
            "quality_client_v2.py",
            "quality-client.sh",
            "quality-client-v2.sh",
            "quality_client_v3.py",
            "quality_client_v3base.py",
            "quality-client-v3.sh",
            "prepare_quality_v3.py",
            "quality-v3-build.json",
            "prepare_quality_controllers.py",
            "prepare_quality_v2.py",
            "quality-controller-build.json",
            "quality-controller-build-v2.json",
            "measure_quality_b0.py",
            "measure_quality_b0_v2.py",
            "measure_quality_microflow.py",
            "measure_quality_microflow_v2.py",
            "measure_quality_kvmat.py",
            "measure_quality_b0_v3.py",
            "measure_quality_microflow_v3.py",
            "measure_quality_kvmat_v3.py",
            "prepare_quality_kvmat.py",
            "quality-kvmat-build.json",
            "recovered-mmlu-cases.json",
            "verified-mmlu-quality-inputs.json",
            "verify_mmlu_quality_inputs.py",
            "recover_quality_wire_inputs.py",
            "quality-wire-inputs.json",
            "quality-wire-source-audit.json",
            "audit_quality_wire_sources.py",
            "test_quality_client.py",
            "audit_quality_pair.py",
            "audit_quality_finishes.py",
            "baseline.py",
            "baseline.sh",
            "runtime-kvmat.sh",
            "launch_kvmat.py",
            "serve-kvmat.sh",
            "runtime-microflow.sh",
            "launch_microflow.py",
            "serve-microflow.sh",
            "microflow-source-receipt.json",
            "microflow-manager.json",
            "microflow-manager-receipt.json",
            "kvmat-source-receipt.json",
            "kvmat-managed-source-receipt.json",
            "materialized.200.jsonl",
            "actual-runtime.json",
        ]
        summaries = {}
    selected = {}
    for directory in directories:
        for path in (args.root / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                selected[path.relative_to(args.root).as_posix()] = path
    for source in sources:
        selected[f"source/{source}"] = args.root / source
    args.output.mkdir(exist_ok=False)
    members = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative, path in sorted(selected.items()):
            members[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=relative, recursive=False)
    payload = buffer.getvalue()
    parts = []
    for index, offset in enumerate(range(0, len(payload), 900_000)):
        part = payload[offset : offset + 900_000]
        name = f"{args.kind}-evidence.tar.gz.part{index:03d}"
        (args.output / name).write_bytes(part)
        parts.append({"file": name, "sha256": hashlib.sha256(part).hexdigest()})
    (args.output / "archive.json").write_text(
        json.dumps(
            {
                "parts": parts,
                "concatenated_sha256": hashlib.sha256(payload).hexdigest(),
                "members": members,
            },
            indent=2,
        )
        + "\n"
    )
    for name, source in summaries.items():
        (args.output / name).write_bytes((args.root / source).read_bytes())


if __name__ == "__main__":
    main()
