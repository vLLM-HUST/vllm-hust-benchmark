"""Acquire the pinned original MMMU prompt and evaluation code; no model calls."""

import argparse
import json
from pathlib import Path

from prepare_chartqa_native_toolkit import acquire_archive, git

REPOSITORY = "MMMU-Benchmark/MMMU"
REVISION = "268471d0d488258990025331c7528359c324aa25"  # pragma: allowlist secret
TREE = "08289f37577cbf1cb6cbf76afae3ecd271a65934"  # pragma: allowlist secret


def verify(root):
    root = Path(root).resolve(strict=True)
    if not (root / ".git").is_dir() or git(root, "write-tree") != TREE:
        raise ValueError("Pinned original MMMU source tree required")
    git(root, "diff", "--exit-code")
    if git(root, "ls-files", "--others"):
        raise ValueError("Untracked original-source files; do not overwrite")
    required = [
        "mmmu/main_parse_and_eval.py",
        "mmmu/utils/data_utils.py",
        "mmmu/utils/eval_utils.py",
        "mmmu/configs/llava1.5.yaml",
    ]
    if not all((root / path).is_file() for path in required):
        raise ValueError("Original MMMU prompt/scorer files missing")
    return {
        "source_id": "MMMU",
        "revision": REVISION,
        "git_tree": TREE,
        "status": "ORIGINAL_SOURCE_VERIFIED_NOT_NATIVE_EXECUTION",
        "model_calls": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = verify(args.output)
    elif args.execute:
        args.output.mkdir(parents=True, exist_ok=False)
        acquire_archive(
            args.output, repository=REPOSITORY, revision=REVISION, tree=TREE
        )
        result = verify(args.output)
    else:
        result = {
            "status": "PLAN_ONLY_NO_DOWNLOAD",
            "repository": REPOSITORY,
            "revision": REVISION,
            "model_calls": 0,
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
