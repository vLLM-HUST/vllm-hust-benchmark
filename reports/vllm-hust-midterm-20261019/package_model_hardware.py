"""Archive selected immutable hardware samples and workload lifecycle evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    selected = {f"run/{p.name}": p for p in args.run.iterdir() if p.is_file()}
    for name in ("sample_microflow_hardware.sh", "measure_microflow.py", "bench.sh"):
        selected[f"launcher/{name}"] = args.run.parent / name
    if args.control:
        for path in args.control.iterdir():
            if path.is_file():
                selected[f"control/{path.name}"] = path
        for name in ("run_model_system_hardware.py", "sample_system_hardware.sh"):
            selected[f"launcher/{name}"] = args.run.parent / name
    for path in args.profile_root.rglob("*"):
        if path.is_file() and path.name.startswith(
            ("hbm", "hccs", "info.json", "start_info", "end_info")
        ):
            selected["profiler/" + path.relative_to(args.profile_root).as_posix()] = (
                path
            )
    members = {}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative, path in sorted(selected.items()):
            members[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=relative, recursive=False)
    payload = buffer.getvalue()
    parts = []
    for i, offset in enumerate(range(0, len(payload), 900_000)):
        part = payload[offset : offset + 900_000]
        name = f"hardware-evidence.tar.gz.part{i:03d}"
        (args.output / name).write_bytes(part)
        parts.append(dict(file=name, sha256=hashlib.sha256(part).hexdigest()))
    archive_info = dict(
        parts=parts,
        concatenated_sha256=hashlib.sha256(payload).hexdigest(),
        members=members,
        excluded="Non-memory profiler task timelines; full original capture remains on the experiment device.",
    )
    (args.output / "archive.json").write_text(json.dumps(archive_info, indent=2) + "\n")
    (args.output / "summary.json").write_bytes(args.summary.read_bytes())
    (args.output / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
            for p in sorted(args.output.iterdir())
            if p.is_file() and p.name != "SHA256SUMS"
        )
    )


if __name__ == "__main__":
    main()
