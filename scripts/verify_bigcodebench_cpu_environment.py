"""Verify a CPU scorer environment without executing tasks or calling a model."""

import argparse
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys


def verify_locked_versions(lock, version_lookup=None):
    from packaging.version import Version

    version_lookup = version_lookup or metadata.version
    versions = {}
    for line in Path(lock).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;]+)", line.strip())
        if not match:
            raise ValueError("exact named distribution versions required")
        name, expected = match.groups()
        key = re.sub(r"[-_.]+", "-", name).lower()
        if key in versions:
            raise ValueError("duplicate locked distribution")
        actual = version_lookup(name)
        if Version(actual) != Version(expected):
            raise ValueError(f"locked distribution mismatch: {name}")
        versions[key] = actual
    if not versions:
        raise ValueError("nonempty CPU environment lock required")
    return versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 10, 22) or platform.machine() != "aarch64":
        raise ValueError("this validated lock requires Python3.10.22/Linux aarch64")
    if sys.platform != "linux" or any(Path("/dev").glob("davinci*")):
        raise ValueError("CPU-only Linux qualification required; no NPU exposure")
    from audit_bigcodebench_scoring_assets import verify_source
    from bigcodebench_container_worker import (
        qualify_author_entrypoints,
        qualify_environment,
    )

    sources = verify_source(args.author_root)
    versions = verify_locked_versions(args.lock)
    original = qualify_environment(
        args.author_root / "Requirements/requirements-eval.txt"
    )
    qualify_author_entrypoints()
    import torch

    if torch.version.cuda is not None or torch.cuda.is_available():
        raise ValueError("CPU scorer unexpectedly uses CUDA")
    print(
        json.dumps(
            {
                "status": "LOCK_AND_ORIGINAL_CLI_QUALIFIED_NOT_TASK_SCORING",
                "locked_distributions_verified": len(versions),
                "original_evaluation_distributions_verified": len(original),
                "original_source_files_verified": len(sources),
                "cpu_torch": torch.__version__,
                "model_calls": 0,
                "task_code_executed": False,
                "deprecated_e2b_warning_retained": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
