"""Prepare fixed original configuration prefixes; never run Agent/environment."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_jsonschemabench_author_source import verify_source
from prepare_multiagentbench_original_assets import LEDGER, prepare as verify_assets

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "task",
    "environment",
    "agents",
    "relationships",
    "metrics",
    "llm",
    "memory",
    "engine_planner",
    "output",
)


def validate_loaded_config(original, loaded):
    for field in FIELDS:
        if getattr(loaded, field) != original[field]:
            raise ValueError(
                "original configuration field changed during native loading"
            )
    if loaded.coordination_mode != original["coordinate_mode"]:
        raise ValueError("original coordination mode changed")


def empty_runtime_fields(original):
    """Expose publisher-cleared bindings without guessing replacements."""
    paths = (
        ("coordinate_mode",),
        ("llm",),
        ("environment", "type"),
        ("environment", "max_iterations"),
        ("memory", "type"),
        ("metrics", "evaluate_llm"),
        ("output", "file_path"),
    )
    empty = []
    for path in paths:
        value = original
        for key in path:
            if not isinstance(value, dict) or key not in value:
                break
            value = value[key]
        else:
            if value is None or value == "":
                empty.append(".".join(path))
    return empty


def prepare(data_root, author_root, output, *, limit=3, execute=False):
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("original prefix must be between1 and100 per file")
    data, author, output = map(
        lambda path: Path(path).resolve(), (data_root, author_root, output)
    )
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("fresh external output directory required")
    if any(
        output.is_relative_to(source) or source.is_relative_to(output)
        for source in (data, author)
    ):
        raise ValueError("output must not overlap original data or software")
    report = {
        "source_id": "MULTIAGENTBENCH",
        "status": "PLAN_ONLY",
        "selection": "First N original rows independently in each of five pinned files; no replacement",
        "per_file_limit": limit,
        "model_calls": 0,
        "native_execution": False,
        "official_scoring": False,
        "source_checkbox_eligible": False,
    }
    if not execute:
        return report
    verify_assets(data, action="verify")
    verify_source(author, ROOT / "configs/workload-native/marble-author-source.json")
    sys.path.insert(0, str(author))
    module = importlib.import_module("marble.configs.config")
    if Path(module.__file__).resolve() != author / "marble/configs/config.py":
        raise ValueError("unexpected author configuration loader")
    files = json.loads(LEDGER.read_text("utf-8"))["delivery_inventory"][
        "MULTIAGENTBENCH"
    ]["protocol_review"]["original_assets"]
    output.mkdir(mode=0o700)
    selected = []
    for item in files:
        originals = [
            json.loads(line)
            for line in (data / item["path"]).read_text("utf-8").splitlines()
            if line.strip()
        ]
        scenario = Path(item["path"]).parent.name
        folder = output / scenario
        folder.mkdir(mode=0o700)
        for index, original in enumerate(originals[:limit]):
            path = folder / f"original-row-{index:05d}.yaml"
            # JSON is accepted by the original yaml.safe_load implementation.
            # No fields, roles, prompts, budgets, model names or paths are rebound.
            raw = json.dumps(original, ensure_ascii=False, indent=2).encode("utf-8")
            with path.open("xb") as stream:
                stream.write(raw)
            if json.loads(path.read_text("utf-8")) != original:
                raise ValueError("original configuration serialization changed")
            validate_loaded_config(original, module.Config.load(str(path)))
            selected.append(
                {
                    "source_file": item["path"],
                    "source_row_index": index,
                    "configuration": path.relative_to(output).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "publisher_cleared_runtime_fields": empty_runtime_fields(original),
                }
            )
    report.update(
        status="FIXED_ORIGINAL_CONFIGURATIONS_NATIVE_LOADER_VERIFIED_NOT_EXECUTED",
        selected_configurations=len(selected),
        configurations=selected,
        original_fields_relations_models_budgets_and_paths_changed=False,
        runtime_bindings_complete=False,
        runtime_boundary="Native Config.load only parses fields. Cleared bindings must be resolved from scenario-specific publisher instructions; do not apply Research/3-iteration converter defaults indiscriminately to all scenarios.",
    )
    (output / "preparation-public.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "author-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = prepare(
        args.data_root,
        args.author_root,
        args.output,
        limit=args.limit,
        execute=args.execute,
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "configurations"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
