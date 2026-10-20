"""Archive HCCS counter qualification and a same-workload diagnostic run."""

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
    parser.add_argument("--kind", choices=("model", "capacity"), default="model")
    args = parser.parse_args()
    selected = {}
    directories = (
        (
            "hccs-known-payload-r2",
            "hccs64-known-payload-control-r2",
            "microflow-hccs64-r1",
            "microflow-hccs64-control-r1",
        )
        if args.kind == "model"
        else ("hccs-capacity-r1", "hccs-capacity-control-r1")
    )
    for directory in directories:
        for path in (args.root / directory).iterdir():
            if path.is_file():
                selected[path.relative_to(args.root).as_posix()] = path
    sources = (
        (
            "dcmi_hccs64.py",
            "hccs_known_payload.py",
            "hccs_known_payload_v2.py",
            "hccs_known_payload_v2.sh",
            "run_hccs64_known_payload.py",
            "qualify_hccs64.py",
            "run_model_hccs64.py",
            "audit_model_hccs64.py",
            "measure_microflow.py",
            "bench.sh",
            "recover_quality_gold.py",
            "recovered-mmlu-gold.json",
        )
        if args.kind == "model"
        else (
            "dcmi_hccs64.py",
            "dcmi_hccs_bandwidth.py",
            "hccs_capacity_probe.py",
            "hccs_capacity_probe.sh",
            "run_hccs_capacity.py",
            "audit_hccs_capacity.py",
            "derive_hccs_link_bytes.py",
            "measure_microflow.py",
        )
    )
    for name in sources:
        selected[f"source/{name}"] = args.root / name
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
        name = f"hccs64-evidence.tar.gz.part{index:03d}"
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
    summaries = (
        (
            (
                "qualification.json",
                "hccs64-known-payload-control-r2/qualification.json",
            ),
            ("summary.json", "microflow-hccs64-control-r1/summary.json"),
        )
        if args.kind == "model"
        else (
            ("summary.json", "hccs-capacity-control-r1/summary.json"),
            (
                "model-link-accounting.json",
                "hccs-capacity-control-r1/model-link-accounting.json",
            ),
        )
    )
    for name, source in summaries:
        (args.output / name).write_bytes((args.root / source).read_bytes())
    (args.output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in sorted(args.output.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        )
    )


if __name__ == "__main__":
    main()
