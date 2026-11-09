"""Original coder Agent in an offline container, with external NPU/tool queues.

This preserves the original prompt and Agent loop. Model-provider configuration
and isolated Python transport are explicit adaptations, not GPT-4 replication.
No scorer or model worker is started by this entrypoint.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from appworld_model_transport import model_proxy
from discoverybench_isolated_python import check_container, TOOL_SHA256
from discoverybench_tool_transport import budgeted_python_run, request_once


AUTHOR_FILES = {
    "agents/coder_utils.py": TOOL_SHA256,
    "agents/coder_agent.py": "1e28ab0169c3170b1c7582ad5f5a10cce2872adc8399a3265cf6be16f0a8c6b6",  # pragma: allowlist secret - original author source
    "utils/autonomous_single_agent.py": "93617028a7547ef26d737fd2a87d289b8b9bb194aaf0d3fd2295a34571cf7d0f",  # pragma: allowlist secret - original author source
    "utils/dv_log.py": "c42a0e45f37ae2a751ed8e02c004c589c26cb6a5084e62ff19c982e11b4f7e34",  # pragma: allowlist secret - original author source
}


def verify_author(source):
    source = Path(source).resolve(strict=True)
    for name, expected in AUTHOR_FILES.items():
        path = (source / name).resolve(strict=True)
        if (
            not path.is_relative_to(source)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"unchanged original Agent source required: {name}")
    return source


def original_queries(value):
    if isinstance(value, list):
        for child in value:
            yield from original_queries(child)
    elif isinstance(value, dict) and "qid" in value and "question" in value:
        yield value
    else:
        raise ValueError("original nested question container required")


def task_inputs(metadata, query_id, dataset_root):
    matches = [
        q for q in original_queries(metadata["queries"]) if int(q["qid"]) == query_id
    ]
    if len(matches) != 1 or not isinstance(matches[0]["question"], str):
        raise ValueError("unique unchanged original question required")
    root = Path(dataset_root).resolve(strict=True)
    paths = []
    for item in metadata["datasets"]:
        relative = Path(item["name"])
        path = (root / relative).resolve(strict=True)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not path.is_relative_to(root)
            or not path.is_file()
        ):
            raise ValueError("original dataset path missing or escaped")
        paths.append(str(path))
    if not paths:
        raise ValueError("original scientific data files required")
    return matches[0]["question"], paths


def execute(args):
    check_container()
    source = verify_author(args.author_root)
    raw = args.metadata.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.metadata_sha256:
        raise ValueError("original metadata identity mismatch")
    metadata = json.loads(raw)
    query, paths = task_inputs(metadata, args.query_id, args.dataset_root)
    for queue in (args.model_queue, args.tool_queue):
        if not queue.is_dir() or any(queue.iterdir()):
            raise ValueError(
                "fresh dedicated queues required; inspect existing runs, never restart"
            )
    args.output.mkdir(mode=0o700, exist_ok=False)
    config = args.output / "model-config-private.json"
    config.write_text(
        json.dumps(
            {"models": {args.model: {"model_type": "openai", "model_name": args.model}}}
        )
    )
    api = args.output / "api-config-private.json"
    api.write_text(json.dumps({"openai": "local-queue-no-secret"}))
    os.environ.update(MODEL_CONFIG=str(config), API_CONFIG=str(api))
    sys.path.insert(0, str(source))
    tools = importlib.import_module("agents.coder_utils")
    original = importlib.import_module("agents.coder_agent")
    workflow = importlib.import_module("utils.autonomous_single_agent")
    if tools.MAX_TURNS != 1:
        raise ValueError("original one-code-call protocol required")
    tools.CustomPythonAstREPLTool._run = budgeted_python_run(
        lambda code: request_once(args.tool_queue, code, timeout=args.transport_timeout)
    )

    class NpuModelProfile(original.BaseAgent):
        def get_model(self, api, api_key, model, **kwargs):
            return super().get_model(
                api=api,
                api_key=api_key,
                model=model,
                max_retries=0,
                request_timeout=args.transport_timeout,
                **kwargs,
            )

    results = []
    with model_proxy(args.model_queue, port=0, timeout=args.transport_timeout) as port:
        os.environ["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        actor = NpuModelProfile(
            model_name=args.model, log_file=str(args.output / "agent-private.log")
        )
        if actor.max_iterations != 5 or actor.agent.max_iterations != 5:
            raise ValueError("original five-iteration Agent protocol required")
        original_log = actor.logger.log_json

        def capture(value):
            results.append(value)
            return original_log(value)

        actor.logger.log_json = capture
        if args.preflight:
            actor.logger.close()
        else:
            workflow.run_autonomous_single_agent_discoverybench(
                agent=actor,
                datasets=paths,
                metadata=metadata,
                nl_query=query,
                provide_domain_knowledge=False,
                provide_workflow_tags=False,
                dataset_type="real",
            )
    if args.metadata.read_bytes() != raw:
        raise ValueError("original metadata changed during execution")
    verify_author(source)
    complete = len(results) == 1 and isinstance(results[0].get("output"), str)
    (args.output / "original-agent-results-private.json").write_text(
        json.dumps(results, indent=2)
    )
    receipt = {
        "source_id": "DISCOVERYBENCH",
        "status": "ORIGINAL_ACTOR_CONSTRUCTED_NOT_RUN"
        if args.preflight
        else "ORIGINAL_ACTOR_RETURNED_NOT_SCORED"
        if complete
        else "ORIGINAL_ACTOR_FAILED_NO_PREDICTION",
        "requested_model": args.model,
        "original_query_id": args.query_id,
        "original_agent_iterations": 5,
        "original_python_call_limit": 1,
        "original_prompt_builder_used": not args.preflight,
        "metadata_sha256": args.metadata_sha256,
        "metadata_unchanged": True,
        "provider_and_python_transport_adaptations": True,
        "official_gpt4_replication": False,
        "native_scoring_completed": False,
        "source_checkbox_eligible": False,
        "actor_construction_only": args.preflight,
    }
    (args.output / "actor-receipt-public.json").write_text(
        json.dumps(receipt, indent=2)
    )
    print(json.dumps(receipt))
    return 0 if complete or args.preflight else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--metadata-sha256")
    parser.add_argument("--query-id", type=int)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--model-queue", type=Path)
    parser.add_argument("--tool-queue", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--transport-timeout", type=float, default=3600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="construct original actor only; no model/tool requests",
    )
    args = parser.parse_args()
    if not args.execute and not args.preflight:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "model_calls": 0,
                    "original_agent_iterations": 5,
                    "original_python_call_limit": 1,
                }
            )
        )
        return 0
    required = (
        "author_root",
        "metadata",
        "metadata_sha256",
        "query_id",
        "dataset_root",
        "model_queue",
        "tool_queue",
        "output",
    )
    if (
        any(getattr(args, name) is None for name in required)
        or args.transport_timeout <= 0
    ):
        parser.error(
            "explicit original assets, private queues/output and positive transport timeout required"
        )
    os.umask(0o077)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
