"""Verify original TemporalWiki software/contracts without importing author code."""

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

from prepare_jsonschemabench_author_source import verify_source

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "configs/workload-native/temporalwiki-author-source.json"


def inspect_contract(author_root):
    pins = json.loads(PINS.read_text("utf-8"))
    root = verify_source(author_root, PINS)
    modes = Counter()
    python_files = 0
    configurations = 0
    for item in pins["files"]:
        path = root / item["path"]
        if path.suffix == ".py":
            ast.parse(path.read_bytes(), filename=item["path"])
            python_files += 1
        if path.suffix == ".json":
            config = json.loads(path.read_bytes())
            if not isinstance(config, dict) or not isinstance(config.get("mode"), str):
                raise ValueError("original configuration mode required")
            modes[config["mode"]] += 1
            configurations += 1
    return {
        "source_id": "TEMPORALWIKI",
        "status": "ORIGINAL_SOFTWARE_AND_CONFIGURATIONS_VERIFIED_NOT_NATIVE_EXECUTION",
        "revision": pins["revision"],
        "files": len(pins["files"]),
        "bytes": sum(item["bytes"] for item in pins["files"]),
        "python_ast_parsed_not_imported": python_files,
        "original_configurations": configurations,
        "configuration_modes": dict(sorted(modes.items())),
        "task_data_acquired": False,
        "legacy_cuda_environment_reproduced": False,
        "model_calls": 0,
        "native_tasks_run": 0,
        "source_checkbox_eligible": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_contract(args.author_root), indent=2))


if __name__ == "__main__":
    main()
