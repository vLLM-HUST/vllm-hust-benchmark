"""Prepare a code-only CPU image context; never build, fetch or execute tasks."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from audit_bigcodebench_scoring_assets import verify_source

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/workload-native"
TRUSTED = (
    "audit_bigcodebench_scoring_assets.py",
    "run_bigcodebench_original_generation.py",
    "bigcodebench_container_worker.py",
    "verify_bigcodebench_cpu_environment.py",
)


def prepare(author, output):
    author = Path(author).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists() or any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (author, ROOT)
    ):
        raise ValueError(
            "fresh external context separate from author/repository required"
        )
    identities = verify_source(author)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    for relative, expected in identities.items():
        source = (author / relative).resolve(strict=True)
        if not source.is_relative_to(author):
            raise ValueError("verified author file escapes source root")
        target = output / "author" / relative
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o644)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError("author changed while preparing context")
    for name in TRUSTED:
        shutil.copyfile(ROOT / "scripts" / name, output / name)
    shutil.copyfile(CONFIG / "bigcodebench-cpu.Dockerfile", output / "Dockerfile")
    shutil.copyfile(
        CONFIG / "bigcodebench-cpu-py310-aarch64.lock.txt",
        output / "requirements.lock.txt",
    )
    return {
        "status": "CODE_ONLY_CPU_CONTEXT_PREPARED_NOT_BUILT",
        "original_source_files_verified": len(identities),
        "repository_runtime_files": len(TRUSTED) + 2,
        "raw_tasks_or_model_weights_copied": False,
        "model_calls": 0,
        "task_code_executed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-code", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    result = (
        prepare(args.official_code, args.output)
        if args.prepare
        else {"status": "PLAN_ONLY", "model_calls": 0, "task_code_executed": False}
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
