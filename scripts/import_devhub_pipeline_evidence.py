"""Preserve the two measured Pipeline records embedded in dev-hub scripts."""

import argparse
import json
from pathlib import Path

from migrate_devhub_evidence import git, sha

FILES = {
    "scripts/frontier_pipeline/RESULTS.json": "results.json",
    "scripts/frontier_pipeline/ecpa/receipt.json": "ecpa-receipt.json",
}


def import_records(source, output):
    if output.exists():
        raise ValueError("Refusing to overwrite preserved evidence")
    commit = git(source, "rev-parse", "origin/main").decode().strip()
    records = []
    payloads = []
    for original, target in FILES.items():
        raw = git(source, "show", f"{commit}:{original}")
        json.loads(raw)
        records.append(
            {
                "source_path": original,
                "destination_path": target,
                "bytes": len(raw),
                "sha256": sha(raw),
            }
        )
        payloads.append((target, raw))
    output.mkdir(parents=True)
    for name, raw in payloads:
        (output / name).write_bytes(raw)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source_repository": "vLLM-HUST/vllm-hust-dev-hub",
                "source_commit": commit,
                "files": records,
                "admission": "Existing measured evidence only; no new measurements or pairing.",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Preserved {len(records)} Pipeline records from {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import_records(args.source, args.output)
