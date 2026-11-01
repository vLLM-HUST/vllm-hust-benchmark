"""Original single-turn SAB baseline using an explicitly named local model.

Preserves original prompts, code extraction, sampling and2000-token budget.
Local NPU API-dollar cost is unspecified (null), never fabricated as zero.
This generates text only; scientific execution belongs in an isolated grader.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace


SOURCE_HASHES = {
    "agent.py": "41552973b8eaae55bb7c78012b37d2e3487570802f01f5bec12b5c39be21e0df",  # pragma: allowlist secret - pinned publisher software
    "run_infer.py": "de02f45825fcfaeb9ca8a3430e6fed1e03436f4071dba5e2e333488b7cb8316a",  # pragma: allowlist secret - pinned publisher software
}
METADATA_SHA256 = "c6f937863a220bd1762a00c20a0f79cc8dfca900b819bdb552150310731ae147"  # pragma: allowlist secret - original verified metadata identity
ENGINE_SHA256 = "ff98f4e7ba786250df5cbbdfd25b103d7dea7637f973da0fe6b4fce50411d569"  # pragma: allowlist secret - existing public transport identity


def original_baseline(source_dir, engine, token_count):
    """Bind three original methods; exclude host-executing debug/install methods."""
    trees = {}
    for name, identity in SOURCE_HASHES.items():
        raw = (Path(source_dir) / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != identity:
            raise ValueError(f"original publisher identity mismatch: {name}")
        trees[name] = ast.parse(raw)
    constants = {
        "SYSTEM_PROMPT",
        "SELF_DEBUG_PROMPT",
        "FORMAT_PROMPT",
        "REQUEST_PROMPT",
        "DATA_INFO_PROMPT",
    }
    nodes = [
        node
        for node in trees["agent.py"].body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in constants
    ]
    agent = next(
        node
        for node in trees["agent.py"].body
        if isinstance(node, ast.ClassDef) and node.name == "ScienceAgent"
    )
    methods = [
        node
        for node in agent.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"get_sys_msg", "write_program", "solve_task"}
    ]
    if len(nodes) != 5 or len(methods) != 3:
        raise ValueError("original baseline method contract changed")
    solve = next(node for node in methods if node.name == "solve_task")
    cost = [
        node
        for node in solve.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "cost"
    ]
    if len(cost) != 1 or not isinstance(cost[0].value, ast.BinOp):
        raise ValueError("one original reporting-only API cost expression required")
    # Only monetary reporting changes: local deployment has no published API rate.
    # No solve/prompt/output/scientific score operation depends on baseline cost.
    cost[0].value = ast.Constant(None)
    agent_class = ast.ClassDef(
        name="BoundBaseline", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    formatter = next(
        node
        for node in trees["run_infer.py"].body
        if isinstance(node, ast.FunctionDef) and node.name == "format_task_dict"
    )

    def require_unchanged_input(messages, model, max_tokens):
        if model != engine.llm_engine_name or max_tokens != 26000:
            raise ValueError("original baseline input budget and actual model required")
        if token_count(messages) > max_tokens:
            raise ValueError(
                "original prompt exceeds local-model budget; not truncated"
            )
        return messages

    namespace = {"Path": Path, "re": re, "trim_messages": require_unchanged_input}
    module = ast.fix_missing_locations(
        ast.Module(body=[*nodes, agent_class, formatter], type_ignores=[])
    )
    exec(compile(module, "<pinned-original-SAB-baseline>", "exec"), namespace)
    obj = namespace["BoundBaseline"]()
    obj.llm_engine = engine
    obj.context_cutoff = 28000
    obj.use_knowledge = obj.use_self_debug = False
    obj.history = []
    obj.sys_msg = ""

    def generate(row, output_path):
        output_path = Path(output_path)
        if output_path.exists():
            raise FileExistsError(
                "retain existing generated program instead of overwriting"
            )
        task = namespace["format_task_dict"](
            row,
            SimpleNamespace(datasets_path="benchmark/datasets/", use_knowledge=False),
        )
        return obj.solve_task(task, str(output_path))

    return generate


def main(args):
    raw = args.metadata.read_bytes()
    if hashlib.sha256(raw).hexdigest() != METADATA_SHA256:
        raise ValueError("unchanged pinned verified Parquet required")
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    rows = pq.read_table(args.metadata).to_pylist()
    if len(rows) != 102:
        raise ValueError("all102 original verified metadata rows required")
    if args.start < 0 or args.limit <= 0 or args.start >= len(rows):
        raise ValueError("valid fixed original selection required")
    selected = rows if args.full else rows[args.start : args.start + args.limit]
    if args.full and (args.start != 0 or args.limit != 3):
        raise ValueError("full and range selection must not be combined")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    transport_path = Path(__file__).with_name("scienceagentbench_local_engine.py")
    transport_raw = transport_path.read_bytes()
    if hashlib.sha256(transport_raw).hexdigest() != ENGINE_SHA256:
        raise ValueError("qualified local transport required")
    spec = importlib.util.spec_from_file_location("sab_local_transport", transport_path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    usage = []
    client = transport.LocalScienceEngine(
        args.model, args.api_base, disable_thinking=True
    )
    if args.model not in {item.id for item in client.client.models.list().data}:
        raise ValueError(
            "actual served model identifier does not match requested model"
        )

    class Engine:
        llm_engine_name = args.model

        def respond(self, messages, temperature, top_p):
            report["active_request_outcome"] = "PENDING_RESPONSE_DO_NOT_BLINDLY_REPLAY"
            receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
            response = client.respond(messages, temperature, top_p)
            saved = (
                args.output_dir
                / f"response-{report['active_original_position']}-private.json"
            )
            with saved.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "text": response[0],
                        "prompt_tokens": response[1],
                        "completion_tokens": response[2],
                    },
                    stream,
                )
            report["active_request_outcome"] = "RESPONSE_SAVED"
            receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if response[2] > 2000:
                raise ValueError(
                    "served output exceeds original requested2000-token budget"
                )
            usage.append(
                {"prompt_tokens": response[1], "completion_tokens": response[2]}
            )
            return response

    def token_count(messages):
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    generate = original_baseline(args.source_dir, Engine(), token_count)
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    report = {
        "source_id": "SCIENCEAGENTBENCH",
        "status": "ORIGINAL_BASELINE_GENERATING",
        "source_checkbox_eligible": False,
        "model": args.model,
        "profile": "Original single-turn/no-knowledge/no-self-debug; local non-thinking binding",
        "output_token_budget": 2000,
        "api_dollar_cost": None,
        "cost_boundary": "Local deployment API-dollar rates unspecified; no original dollar-cost reproduction claimed",
        "generated_code_executed": False,
        "results": [],
    }
    receipt = args.output_dir / "receipt-public.json"
    try:
        for index, row in enumerate(selected):
            name = row["gold_program_name"]
            if (
                not isinstance(name, str)
                or Path(name).name != name
                or not name.endswith(".py")
                or "\\" in name
            ):
                raise ValueError("simple original Python program filename required")
            program = args.output_dir / ("pred_" + name)
            report["active_original_position"] = (
                index if args.full else args.start + index
            )
            report["active_request_outcome"] = "NOT_SENT"
            receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
            trajectory = generate(row, program)
            path = args.output_dir / f"trajectory-{row['instance_id']}-private.json"
            path.write_text(json.dumps(trajectory), encoding="utf-8")
            report["results"].append(
                {
                    "original_position": index if args.full else args.start + index,
                    "instance_id": str(row["instance_id"]),
                    "program": program.name,
                    "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
                    "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "usage": usage[-1],
                }
            )
            receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["active_original_position"] = None
        report["status"] = (
            "ORIGINAL_BASELINE_GENERATED_ISOLATED_EXECUTION_AND_SCORING_PENDING"
        )
    except BaseException as exc:
        report.update(
            status="GENERATION_FAILED_RETAINED_DO_NOT_REPLACE_TASK",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    main(parser.parse_args())
