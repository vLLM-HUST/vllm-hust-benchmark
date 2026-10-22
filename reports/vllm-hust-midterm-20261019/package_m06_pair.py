"""Import immutable M06 observations and record portable evidence hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("archives", type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / "m06-paired-20261021"
    destination.mkdir(exist_ok=False)
    names = [
        "m06-prechecked-pair-result-r1.json",
        "m06-prechecked-pair-plan-r1.json",
        "m06-pair-result-r2.json",
        "m06-shutdown-boundary-audit-20261021.json",
        "manager-shutdown-envelope-r1.json",
        "audit_m06_prechecked_pair_r1.py",
        "audit_m06_pair_r1.py",
        "test_m06_accounting_r1.py",
    ]
    archives = [
        "m06-prechecked-b0-r1-evidence.tar.gz",
        "m06-prechecked-b1-r1-evidence.tar.gz",
        "m06-pair-b0-r1-evidence.tar.gz",
        "m06-pair-b1-r1-evidence.tar.gz",
    ]
    files = []
    for directory, name in [(args.source, n) for n in names] + [
        (args.archives, n) for n in archives
    ]:
        source = directory / name
        shutil.copyfile(source, destination / name)
        files.append(
            {
                "name": name,
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    index = {
        "schema": "m06-paired-evidence/v1",
        "release": "https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/m01-recovery-20261021-r7",
        "files": files,
        "original_evidence_modified": False,
        "os_page_cache_globally_flushed": False,
        "independent_services_and_empty_compile_caches": True,
    }
    (destination / "archive.json").write_text(json.dumps(index, indent=2) + "\n")


if __name__ == "__main__":
    main()
