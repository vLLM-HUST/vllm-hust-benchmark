"""Verify core software only; never load tasks or execute generated code."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def verify(lock):
    if sys.version_info[:3] != (3, 11, 17) or platform.machine() != "aarch64":
        raise ValueError("pinned Linux ARM CPython3.11.17 core image required")
    expected = {}
    for line in Path(lock).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError("exact observed version pins required")
        name, version = line.split("==")
        if name in expected:
            raise ValueError("duplicate observed package")
        expected[name] = version
    if len(expected) != 83 or expected.get("appworld") != "0.1.3.post1":
        raise ValueError("complete observed83 core resolution required")
    for name, version in expected.items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"core software version differs: {name}")
    from appworld.environment import AppWorld
    from appworld.evaluator import evaluate_task

    if not callable(AppWorld) or not callable(evaluate_task):
        raise ValueError("original core environment/evaluator imports unavailable")
    return {
        "status": "CORE_SOFTWARE_VERIFIED_NOT_NATIVE_TASK_EXECUTION",
        "observed_packages_verified": len(expected),
        "model_calls": 0,
        "task_loaded": False,
        "agent_executed": False,
        "scoring_executed": False,
        "source_checkbox_eligible": False,
        "boundary": "Observed core resolution, not an author lock or task isolation qualification",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.lock), indent=2))
