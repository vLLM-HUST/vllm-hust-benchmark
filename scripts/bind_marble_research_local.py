"""Bind existing official Research prefixes to explicitly named local models.

Preparation only: no Agent, model, tool, network or scoring execution. Original
tasks, agents, relationships, prompts, five iterations and unknown fields stay
unchanged. Actor/judge substitution is disclosed, not official-model parity.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re


def model_name(value):
    if not re.fullmatch(r"openai/[A-Za-z0-9][A-Za-z0-9._/-]*", value):
        raise ValueError("explicit local OpenAI-compatible model name required")
    return value


def bind_config(original, *, model, judge, result_path):
    model_name(model)
    model_name(judge)
    if (
        original.get("environment", {}).get("type") != "Research"
        or original["environment"].get("max_iterations") != 5
        or original.get("coordinate_mode") != "graph"
        or original.get("memory", {}).get("type") != "BaseMemory"
        or not original.get("agents")
        or any("llm" in agent for agent in original["agents"])
        or original.get("llm") != "gpt-3.5-turbo"
        or original.get("metrics", {}).get("evaluate_llm") != "gpt-4o"
        or not isinstance(original.get("output"), dict)
    ):
        raise ValueError(
            "verified original Research profile with inherited model required"
        )
    result = deepcopy(original)
    result["llm"] = model
    result["metrics"]["evaluate_llm"] = judge
    result["output"]["file_path"] = str(result_path)
    # Restore only the three explicit deployment bindings and compare all else.
    audit = deepcopy(result)
    audit["llm"] = original["llm"]
    audit["metrics"]["evaluate_llm"] = original["metrics"]["evaluate_llm"]
    if "file_path" in original["output"]:
        audit["output"]["file_path"] = original["output"]["file_path"]
    else:
        audit["output"].pop("file_path")
    if audit != original:
        raise ValueError("original task protocol changed beyond deployment bindings")
    return result


def prepare(prepared_root, output, *, model, judge, execute=False):
    model_name(model)
    model_name(judge)
    source, output = Path(prepared_root).absolute(), Path(output).absolute()
    if output.exists() or source == output or output.is_relative_to(source):
        raise ValueError("fresh separate private output directory required")
    report = {
        "source_id": "MULTIAGENTBENCH",
        "status": "PLAN_ONLY",
        "actor_model": model,
        "judge_model": judge,
        "official_actor_model": "gpt-3.5-turbo",
        "official_judge_model": "gpt-4o",
        "official_model_reproduction": False,
        "model_calls": 0,
        "native_execution": False,
        "source_checkbox_eligible": False,
    }
    if not execute:
        return report
    if source.is_symlink():
        raise ValueError("non-symlink original preparation required")
    source, output = source.resolve(), output.resolve()
    if (
        output.is_relative_to(source)
        or source.is_relative_to(output)
        or output.is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise ValueError("private output must not overlap source or repository")
    receipt = json.loads((source / "preparation-public.json").read_text("utf-8"))
    if (
        receipt.get("source_id") != "MULTIAGENTBENCH"
        or receipt.get("status")
        != "OFFICIAL_RESEARCH_RUNTIME_PROFILE_PREPARED_NOT_EXECUTED"
        or receipt.get("runtime_profile") != "official-research"
    ):
        raise ValueError("original official Research preparation receipt required")
    entries = receipt["configurations"]
    if (
        not entries
        or any(type(item["source_row_index"]) is not int for item in entries)
        or len(entries) != receipt["selected_configurations"]
        or [item["source_row_index"] for item in entries] != list(range(len(entries)))
        or any(
            item["source_file"] != "multiagentbench/research/research_main.jsonl"
            for item in entries
        )
    ):
        raise ValueError("complete unchanged original Research prefix required")
    configs = []
    for item in entries:
        path = source / item["configuration"]
        if path.is_symlink() or not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("configuration escapes original preparation")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("prepared original configuration identity mismatch")
        identity = item["source_row_index"]
        result = bind_config(
            json.loads(raw),
            model=model,
            judge=judge,
            result_path=output / f"original-row-{identity:05d}-result.jsonl",
        )
        configs.append((identity, result, item["sha256"]))
    output.mkdir(mode=0o700)
    bindings = []
    for identity, config, prior in configs:
        path = output / f"original-row-{identity:05d}.yaml"
        raw = json.dumps(config, ensure_ascii=False, indent=2).encode()
        with path.open("xb") as stream:
            stream.write(raw)
        bindings.append(
            {
                "source_row_index": identity,
                "configuration": path.name,
                "original_preparation_sha256": prior,
                "bound_configuration_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    report.update(
        status="LOCAL_MODELS_BOUND_ORIGINAL_RESEARCH_PROTOCOL_NOT_EXECUTED",
        selected_configurations=len(bindings),
        configurations=bindings,
        changed_fields=["llm", "metrics.evaluate_llm", "output.file_path"],
        original_task_agents_relationships_and_unknown_fields_preserved=True,
        original_iterations=5,
    )
    (output / "binding-public.json").write_text(json.dumps(report, indent=2), "utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = prepare(
        args.prepared_root,
        args.output,
        model=args.model,
        judge=args.judge,
        execute=args.execute,
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "configurations"}, indent=2)
    )


if __name__ == "__main__":
    main()
