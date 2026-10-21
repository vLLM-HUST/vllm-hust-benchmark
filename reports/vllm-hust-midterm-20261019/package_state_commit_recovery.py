"""Package recovery diagnostics and unchanged-source numeric evidence."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = args.source.resolve()
    destination = Path(__file__).resolve().parent / "m01-recovery/state-commit-recovery"
    destination.mkdir(exist_ok=False)
    names = set()
    for name, key in [
        ("runtime-import-recovery-audit-v4.json", "files"),
        ("state-commit-recovery-audit-r3.json", "source_sha256"),
    ]:
        audit = json.loads((root / name).read_text())
        names.add(name)
        for relative, digest in audit[key].items():
            path = (root / relative).resolve()
            assert path.is_relative_to(root)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            names.add(relative)
    names.update(
        [
            "probe_triton_candidates_v1.py",
            "probe_triton_candidates_v1.sh",
            "test_triton_candidate_record.py",
        ]
    )
    names.update(
        str(path.relative_to(root))
        for path in (root / "triton-candidate-import-r1").iterdir()
    )
    archive_path = destination / "evidence.tar.gz"
    members = []
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in sorted(names):
            path = root / name
            assert path.is_file() and not path.is_symlink()
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
            members.append(
                dict(name=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())
            )
    packed = archive_path.read_bytes()
    index = dict(
        archive="evidence.tar.gz",
        archive_bytes=len(packed),
        archive_sha256=hashlib.sha256(packed).hexdigest(),
        members=members,
        model_qualification=False,
        performance_publication=False,
    )
    (destination / "archive.json").write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps(dict(members=len(members), bytes=len(packed))))


if __name__ == "__main__":
    main()
