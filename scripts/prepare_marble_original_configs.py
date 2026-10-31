"""Prepare original prefixes or an explicit publisher Research runtime profile.

Never run Agent/environment; default preserves every original field unchanged.
"""

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import re
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


def official_research_defaults(author):
    """Read pinned literal publisher bindings; never execute its shell script."""
    text = (author / "multiagentbench/runjsonl2yaml.sh").read_text("utf-8")
    values = {}
    for name in (
        "INPUT_FILE",
        "DEFAULT_COORDINATE_MODE",
        "DEFAULT_ENVIRONMENT",
        "DEFAULT_LLM",
        "DEFAULT_MEMORY",
        "DEFAULT_METRICS_EVALUATE_LLM",
        "DEFAULT_OUTPUT",
    ):
        match = re.search(r"^" + name + r"=([\"'])(.*?)\1$", text, re.MULTILINE)
        if match is None:
            raise ValueError("pinned publisher literal binding missing")
        values[name] = match.group(2)
    environment = json.loads(values["DEFAULT_ENVIRONMENT"])
    if (
        values["INPUT_FILE"] != "./research/research_main.jsonl"
        or environment["type"] != "Research"
        or environment["max_iterations"] != 5
    ):
        raise ValueError("publisher Research profile mismatch")
    return {
        "coordinate_mode": values["DEFAULT_COORDINATE_MODE"],
        "environment": environment,
        "llm": values["DEFAULT_LLM"],
        "memory": json.loads(values["DEFAULT_MEMORY"]),
        "metrics": {"evaluate_llm": values["DEFAULT_METRICS_EVALUATE_LLM"]},
        "output": json.loads(values["DEFAULT_OUTPUT"]),
    }


def prepare(
    data_root, author_root, output, *, limit=3, execute=False, runtime_profile=None
):
    if runtime_profile not in (None, "official-research"):
        raise ValueError("only the verified official Research profile is supported")
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
        "selection": "First N original Research rows only; publisher literal runtime profile"
        if runtime_profile
        else "First N original rows independently in each of five pinned files; no replacement",
        "runtime_profile": runtime_profile,
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
    defaults = official_research_defaults(author) if runtime_profile else None
    converter = None
    if runtime_profile:
        spec = importlib.util.spec_from_file_location(
            "pinned_marble_converter", author / "multiagentbench/jsonl2yaml.py"
        )
        converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(converter)
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
        if runtime_profile and scenario != "research":
            continue
        folder = output / scenario
        folder.mkdir(mode=0o700)
        for index, original in enumerate(originals[:limit]):
            runtime = (
                converter.fill_defaults(copy.deepcopy(original), defaults)
                if converter
                else original
            )
            path = folder / f"original-row-{index:05d}.yaml"
            # JSON is accepted by the original yaml.safe_load implementation.
            # Default preserves all fields. Explicit Research mode fills only
            # publisher-cleared runtime slots through its original converter.
            raw = json.dumps(runtime, ensure_ascii=False, indent=2).encode("utf-8")
            with path.open("xb") as stream:
                stream.write(raw)
            if json.loads(path.read_text("utf-8")) != runtime:
                raise ValueError("original configuration serialization changed")
            validate_loaded_config(runtime, module.Config.load(str(path)))
            for key in ("task", "agents", "relationships"):
                if runtime[key] != original[key]:
                    raise ValueError(
                        "publisher binding changed original task/agents/relations"
                    )
            selected.append(
                {
                    "source_file": item["path"],
                    "source_row_index": index,
                    "configuration": path.relative_to(output).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "publisher_cleared_runtime_fields": empty_runtime_fields(original),
                    "runtime_profile": runtime_profile,
                }
            )
    report.update(
        status="OFFICIAL_RESEARCH_RUNTIME_PROFILE_PREPARED_NOT_EXECUTED"
        if runtime_profile
        else "FIXED_ORIGINAL_CONFIGURATIONS_NATIVE_LOADER_VERIFIED_NOT_EXECUTED",
        selected_configurations=len(selected),
        configurations=selected,
        original_fields_relations_models_budgets_and_paths_changed=bool(
            runtime_profile
        ),
        original_task_agents_and_relations_changed=False,
        publisher_cleared_runtime_bindings_filled=bool(runtime_profile),
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
    parser.add_argument("--runtime-profile", choices=["official-research"])
    args = parser.parse_args()
    report = prepare(
        args.data_root,
        args.author_root,
        args.output,
        limit=args.limit,
        execute=args.execute,
        runtime_profile=args.runtime_profile,
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "configurations"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
