"""Inspect pinned original PaperBench prerequisites without installing or building.

This is an offline preflight, not native Agent, reproduction or grading evidence.
It never accepts vendor terms, downloads images or invokes author code.
"""

import argparse
import json
from pathlib import Path
import platform
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_crosscodeeval_original_assets import verify_git_blob
from prepare_xsum_original_assets import bound_path


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/paperbench-author-source.json"
)
PROJECT = "project/paperbench/"
RECIPES = [
    PROJECT + "paperbench/Dockerfile.base",
    PROJECT + "paperbench/reproducer.Dockerfile",
    PROJECT + "paperbench/scripts/build-docker-images.sh",
    PROJECT + "paperbench/solvers/basicagent/solver.py",
    PROJECT + "README.md",
]


def inspect_original(author_root, *, pins_path=PINS, host_architecture=None):
    pins = json.loads(Path(pins_path).read_text("utf-8"))
    if pins.get("source_id") != "PAPERBENCH":
        raise ValueError("pinned PaperBench software profile required")
    root = Path(author_root).resolve(strict=True)
    files = pins["files"]
    if len({item["path"] for item in files}) != len(files):
        raise ValueError("duplicate pinned source path")
    originals = {}
    for item in files:
        path = bound_path(root, "", item)
        verify_git_blob(path, item)
        if item["path"] in RECIPES:
            originals[item["path"]] = path.read_text("utf-8")
    if set(originals) != set(RECIPES):
        raise ValueError("original Docker, solver and README pins required")
    build = originals[RECIPES[2]]
    base = originals[RECIPES[0]]
    if build.count("--platform=linux/amd64") != 2:
        raise ValueError("review changed original Docker platform contract")
    if "Miniconda3-py313_25.5.1-0-Linux-x86_64.sh" not in base:
        raise ValueError("review changed original Miniconda contract")
    if "conda tos accept" not in base:
        raise ValueError("review changed original vendor-terms behavior")
    if "time_limit: int | None = chz.field(default=5 * 60" not in originals[RECIPES[3]]:
        raise ValueError("review changed original BasicAgent time budget")
    if "paperbench.reproduction.timeout=60" not in originals[RECIPES[4]]:
        raise ValueError("review changed author development protocol")
    architecture = host_architecture or platform.machine()
    return {
        "source_id": "PAPERBENCH",
        "status": "PINNED_ORIGINAL_PREREQUISITES_INSPECTED_NOT_NATIVE_READY",
        "source_revision": pins["revision"],
        "verified_original_files": len(files),
        "host_architecture": architecture,
        "original_image_platform": "linux/amd64",
        "native_host_architecture_matches": architecture.lower() in {"amd64", "x86_64"},
        "cross_arch_execution_qualified": False,
        "original_recipe_auto_accepts_anaconda_terms": True,
        "vendor_terms_accepted_by_this_check": False,
        "original_basicagent_default_time_seconds": 300,
        "author_documented_dev_reproduction_timeout_seconds": 60,
        "default_and_dev_are_not_canonical_24h_protocol": True,
        "docker_built_or_run": False,
        "model_calls": 0,
        "native_agent_reproduction_or_scoring_executed": False,
        "publisher_reproduction_entry": PROJECT + "paperbench/reproduce.py",
        "publisher_scoring_entry": PROJECT + "paperbench/grade.py",
        "scoring_input_contract": "Reproduction submission archive and matching original paper rubric",
        "source_checkbox_eligible": False,
        "remaining": [
            "Qualify actual isolated images and execution architecture; matching host alone is insufficient",
            "Resolve vendor terms before invoking the original build recipe",
            "Bind suitable models and preserve the complete declared Agent/reproduction/judge protocol",
            "Validate original tasks and official scoring; debug dummy runs are not delivery evidence",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--author-root",
        type=Path,
        help="verified source root containing project/paperbench",
    )
    args = parser.parse_args()
    if args.author_root is None:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "source_id": "PAPERBENCH",
                    "source_profile": str(PINS),
                    "model_calls": 0,
                    "author_code_executed": False,
                },
                indent=2,
            )
        )
        return
    print(json.dumps(inspect_original(args.author_root), indent=2))


if __name__ == "__main__":
    main()
