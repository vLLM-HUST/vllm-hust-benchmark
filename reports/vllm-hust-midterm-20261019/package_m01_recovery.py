"""Package immutable counter qualification artifacts, not full model acceptance."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.evidence_root, args.output
    out.mkdir(exist_ok=False)
    selected = set()
    for folder in (
        "memory-counter-v2",
        "memory-counter-v2-control",
        "hbm-sampling-probe",
        "hbm-sampling-control",
    ):
        selected.update(p for p in (root / folder).rglob("*") if p.is_file())
    for name in (
        "memory_counter_probe_v2.py",
        "memory_counter_probe_v2.sh",
        "run_memory_counter_v2.py",
        "hbm_sampling_probe.py",
        "hbm_sampling_probe.sh",
        "run_hbm_sampling_probe.py",
        "measure_pair.py",
    ):
        selected.add(root / name)
    for path in (root / "hbm-sampling-profiler").rglob("*"):
        if path.is_file() and (
            path.name.startswith(("hbm", "hccs", "info.json", "start_info", "end_info"))
        ):
            selected.add(path)
    manifest = {}
    with tarfile.open(out / "counter-qualification.tar.gz", "w:gz") as archive:
        for path in sorted(selected):
            if not path.is_file():
                raise FileNotFoundError(path)
            relative = path.relative_to(root).as_posix()
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=relative, recursive=False)
    (out / "archive-members.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for src, dst in (
        ("recovered-profile-accounting-v2.json", "recovered-profile-accounting.json"),
        ("hbm-sampling-integration-v2.json", "hbm-sampling-integration.json"),
    ):
        shutil.copyfile(root / src, out / dst)
    manifest = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.iterdir())
        if p.is_file()
    }
    (out / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in manifest.items())
    )


if __name__ == "__main__":
    main()
