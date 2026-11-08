"""Inspect pinned original PaperBench prerequisites without installing or building.

This is an offline preflight, not native Agent, reproduction or grading evidence.
It never accepts vendor terms, downloads images or invokes author code.
"""

import argparse
import ast
import json
from pathlib import Path
import platform
import re
import shlex
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_crosscodeeval_original_assets import verify_git_blob
from prepare_paperbench_original_assets import load_pins as load_paper_asset_pins
from prepare_xsum_original_assets import bound_path


PINS = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/paperbench-author-source.json"
)
PROJECT = "project/paperbench/"
ALL_SPLIT = PROJECT + "experiments/splits/all.txt"
NANOEVAL = "project/common/nanoeval/nanoeval/"
COMPLETERS = "project/common/preparedness_turn_completer/preparedness_turn_completer/"
RECIPES = [
    PROJECT + "paperbench/Dockerfile.base",
    PROJECT + "paperbench/reproducer.Dockerfile",
    PROJECT + "paperbench/scripts/build-docker-images.sh",
    PROJECT + "paperbench/solvers/basicagent/solver.py",
    PROJECT + "README.md",
    PROJECT + "paperbench/nano/eval.py",
    PROJECT + "paperbench/nano/structs.py",
]


def fixed_first_paper(split_text, expected_count=20):
    """Select the publisher's first full-split paper before any model result."""
    ids = [line.strip() for line in split_text.splitlines() if line.strip()]
    if (
        len(ids) != expected_count
        or len(set(ids)) != expected_count
        or any(re.fullmatch(r"[a-z0-9][a-z0-9-]*", paper) is None for paper in ids)
    ):
        raise ValueError("invalid original PaperBench full split")
    return ids[0], len(ids)


def author_fixed_sample_options(runner_args, runner_execution):
    """Require native prefix selection; never rewrite the publisher's split file."""
    slice_statement = "tasks = tasks[: spec.runner.n_tasks]"
    ordered_statements = (
        "tasks = list(await spec.eval.get_tasks())",
        slice_statement,
        "random.shuffle(tasks)",
    )
    if (
        "n_tasks: int | None = chz.field(" not in runner_args
        or "shuffle: bool = False" not in runner_args
        or any(
            runner_execution.count(statement) != 1 for statement in ordered_statements
        )
        or sorted(runner_execution.index(statement) for statement in ordered_statements)
        != [runner_execution.index(statement) for statement in ordered_statements]
    ):
        raise ValueError("review changed author NanoEval prefix-selection contract")
    return {
        "paperbench.paper_split": "all",
        "runner.n_tasks": 1,
        "runner.shuffle": False,
        "runner.concurrency": 1,
        "runner.should_backup": False,
        "runner.enable_slackbot": False,
    }


def inspect_fixed_paper(author_root, *, source_pins=PINS, asset_pins=None):
    """Verify original full-split position and all pinned files of the chosen paper."""
    root = Path(author_root).resolve(strict=True)
    source = json.loads(Path(source_pins).read_text("utf-8"))
    entries = [item for item in source["files"] if item["path"] == ALL_SPLIT]
    if source["source_id"] != "PAPERBENCH" or len(entries) != 1:
        raise ValueError("unique pinned PaperBench full split required")
    split = bound_path(root, "", entries[0])
    verify_git_blob(split, entries[0])
    selected, full_count = fixed_first_paper(split.read_text("utf-8"))
    assets = (
        load_paper_asset_pins()
        if asset_pins is None
        else load_paper_asset_pins(asset_pins)
    )
    if source["revision"] != assets["revision"]:
        raise ValueError("PaperBench source and assets must use the same revision")
    prefix = PROJECT + "data/papers/" + selected + "/"
    files = [item for item in assets["files"] if item["path"].startswith(prefix)]
    required = {"paper.pdf", "paper.md", "config.yaml", "rubric.json"}
    if not required <= {item["path"][len(prefix) :] for item in files}:
        raise ValueError(
            "fixed paper, registry configuration and original rubric required"
        )
    for item in files:
        verify_git_blob(bound_path(root, "", item), item)
    runner_sources = []
    for name in ("eval.py", "evaluation.py"):
        pinned = [item for item in source["files"] if item["path"] == NANOEVAL + name]
        if len(pinned) != 1:
            raise ValueError("unique fixed NanoEval runner source required")
        path = bound_path(root, "", pinned[0])
        verify_git_blob(path, pinned[0])
        runner_sources.append(path.read_text("utf-8"))
    return {
        "status": "FIXED_ORIGINAL_FULL_SPLIT_PAPER_ASSETS_VERIFIED_NOT_RUN",
        "publisher_split": "all",
        "publisher_split_papers": full_count,
        "sample_paper_id": selected,
        "selection_rule": "first nonempty entry in the pinned original all.txt",
        "verified_original_paper_files": len(files),
        "verified_original_paper_bytes": sum(item["bytes"] for item in files),
        "author_sample_runner_overrides": author_fixed_sample_options(*runner_sources),
        "publisher_split_unchanged": True,
        "full_split_aggregate_eligible": False,
        "native_agent_reproduction_or_scoring_executed": False,
        "source_checkbox_eligible": False,
    }


def canonical_protocol(readme):
    """Read, but never run, the publisher's fixed canonical command."""
    section = "### Canonical command\n"
    if readme.count(section) != 1:
        raise ValueError("review changed author canonical protocol section")
    command = readme.split(section, 1)[1].split("### ", 1)[0].split("```")
    if len(command) != 3 or not command[1].startswith("bash\n"):
        raise ValueError("review changed author canonical command block")
    tokens = shlex.split(command[1][5:].replace("\\\n", " "))
    if tokens[:4] != ["uv", "run", "python", "-m"] or tokens[4:5] != [
        "paperbench.nano.entrypoint"
    ]:
        raise ValueError("review changed author canonical entrypoint")
    options = {}
    for token in tokens[5:]:
        key, equals, value = token.partition("=")
        if not equals or key in options:
            raise ValueError("invalid author canonical argument")
        options[key] = value
    required = {
        "paperbench.paper_split": "all",
        "paperbench.solver": "paperbench.solvers.basicagent.solver:BasicAgentSolver",
        "paperbench.solver.completer_config": (
            "paperbench.solvers.basicagent.completer:OpenAIResponsesTurnCompleterConfig"
        ),
        "paperbench.solver.completer_config.tools": "[{'type':'web_search'}]",
        "paperbench.solver.time_limit": "86400",
        "paperbench.reproduction.timeout": "86400",
        "runner.max_retries": "0",
    }
    if any(options.get(key) != value for key, value in required.items()):
        raise ValueError("review changed author canonical protocol options")
    return {
        "paper_split": options["paperbench.paper_split"],
        "agent_time_limit_seconds": int(options["paperbench.solver.time_limit"]),
        "reproduction_timeout_seconds": int(options["paperbench.reproduction.timeout"]),
        "responses_builtin_tools": ["web_search"],
        "runner_max_retries": int(options["runner.max_retries"]),
    }


def inspect_vllm_browser_dispatch(package_root, canonical):
    """Compare the author tool name with vLLM's browser filter, without imports."""
    responses = Path(package_root) / "entrypoints/openai/responses"
    serving = ast.parse((responses / "serving.py").read_text("utf-8"))
    helpers = ast.parse((responses / "utils.py").read_text("utf-8"))
    recognized = set()
    for node in ast.walk(serving):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.BoolOp):
            continue
        appends_browser = any(
            ast.unparse(child) == "builtin_tool_list.append('browser')"
            for child in node.body
        )
        has_browser = any(
            ast.unparse(part) == "self.tool_server.has_tool('browser')"
            for part in node.test.values
        )
        if (
            not appends_browser
            or not has_browser
            or not isinstance(node.test.op, ast.And)
        ):
            continue
        for part in node.test.values:
            if (
                isinstance(part, ast.Compare)
                and isinstance(part.left, ast.Constant)
                and isinstance(part.left.value, str)
                and len(part.ops) == 1
                and isinstance(part.ops[0], ast.In)
                and ast.unparse(part.comparators[0]) == "requested_tool_types"
            ):
                recognized.add(part.left.value)
    binding = any(
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and ast.unparse(node.targets[0]) == "requested_tool_types"
        and ast.unparse(node.value) == "extract_tool_types(request.tools)"
        for node in ast.walk(serving)
    )
    extractor = next(
        (
            node
            for node in helpers.body
            if isinstance(node, ast.FunctionDef) and node.name == "extract_tool_types"
        ),
        None,
    )
    literal_types = extractor is not None and any(
        isinstance(node, ast.If)
        and ast.unparse(node.test) == "tool.type == 'mcp'"
        and [ast.unparse(child) for child in node.orelse]
        == ["tool_types.add(tool.type)"]
        for node in ast.walk(extractor)
    )
    requested = set(canonical["responses_builtin_tools"])
    known_filter = bool(recognized and binding and literal_types)
    if not known_filter:
        status = "STATIC_DISPATCH_SHAPE_UNRECOGNIZED_MANUAL_REVIEW_REQUIRED"
    elif requested.isdisjoint(recognized):
        status = "CANONICAL_TOOL_TYPE_ABSENT_FROM_STATIC_BROWSER_FILTER"
    else:
        status = "CANONICAL_TOOL_TYPE_PRESENT_STATIC_ONLY"
    return {
        "status": status,
        "canonical_tool_types": sorted(requested),
        "observed_browser_filter_types": sorted(recognized),
        "literal_non_mcp_type_extraction_observed": literal_types,
        "model_or_search_calls": 0,
        "runtime_tool_execution_qualified": False,
    }


def author_model_routes(responses_client, judge_client, judge_config):
    """Identify the fixed author model API paths without instantiating clients."""
    if (
        "return openai.AsyncClient()" not in responses_client
        or "return openai.AsyncClient()" not in judge_client
        or "self._client.responses.parse(" not in responses_client
        or "self._client.chat.completions.parse(" not in judge_client
    ):
        raise ValueError("review changed author model-client routing")
    config = judge_config.split("class JudgeConfig(", 1)[-1].split("\nclass ", 1)[0]
    if any(
        value not in config
        for value in (
            "grade_locally: bool = True",
            'scaffold: str = "simple"',
            'model="o3-mini-2025-01-31"',
            'reasoning_effort="high"',
        )
    ):
        raise ValueError("review changed author default local judge profile")
    return {
        "actor_api": "/v1/responses",
        "default_local_judge_api": "/v1/chat/completions",
        "default_local_judge_model": "o3-mini-2025-01-31",
        "default_local_judge_reasoning_effort": "high",
        "default_local_judge_scaffold": "simple",
        "clients_constructed_without_explicit_base_urls": True,
        "actor_and_default_local_judge_need_shared_process_endpoint_binding": True,
        "runtime_model_or_tool_calls": 0,
    }


def inspect_author_model_routes(author_root, *, pins_path=PINS):
    source = json.loads(Path(pins_path).read_text("utf-8"))
    if source["source_id"] != "PAPERBENCH":
        raise ValueError("pinned PaperBench software profile required")
    paths = (
        COMPLETERS + "oai_responses_turn_completer/completer.py",
        COMPLETERS + "oai_completions_turn_completer.py",
        PROJECT + "paperbench/nano/structs.py",
    )
    text = []
    for name in paths:
        entries = [item for item in source["files"] if item["path"] == name]
        if len(entries) != 1:
            raise ValueError("unique pinned author model routing source required")
        path = bound_path(Path(author_root), "", entries[0])
        verify_git_blob(path, entries[0])
        text.append(path.read_text("utf-8"))
    return author_model_routes(*text)


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
    solver = originals[RECIPES[3]]
    if (
        "OpenAIResponsesTurnCompleterConfig(" not in solver
        or 'tools=[WebSearchToolParam(type="web_search_preview")]' not in solver
    ):
        raise ValueError("review changed original BasicAgent model/tool contract")
    if "paperbench.reproduction.timeout=60" not in originals[RECIPES[4]]:
        raise ValueError("review changed author development protocol")
    canonical = canonical_protocol(originals[RECIPES[4]])
    evaluation, structs = (originals[name] for name in RECIPES[5:])
    if (
        'default="pb-env:latest"' not in evaluation
        or 'docker_image="pb-reproducer:latest"' not in structs
        or 'docker_image="pb-env:latest"' not in structs
        or "grade_locally: bool = True" not in structs
    ):
        raise ValueError(
            "review changed original Agent, reproducer or judge image contract"
        )
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
        "original_image_roles": {
            "agent": "pb-env:latest",
            "reproduction": "pb-reproducer:latest",
            "remote_judge": "pb-env:latest",
            "judge_defaults_to_local": True,
        },
        "vendor_terms_accepted_by_this_check": False,
        "original_basicagent_default_time_seconds": 300,
        "original_basicagent_default_model_api": "OpenAI Responses",
        "original_basicagent_default_builtin_tools": ["web_search_preview"],
        "model_endpoint_and_tool_execution_qualified": False,
        "chat_completions_health_check_is_sufficient": False,
        "author_documented_dev_reproduction_timeout_seconds": 60,
        "original_canonical_protocol": canonical,
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
            "Qualify Responses API and original tool execution; a chat-completions endpoint alone is insufficient",
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
    parser.add_argument(
        "--vllm-source",
        type=Path,
        help="optional vllm Python package source root for offline browser dispatch inspection",
    )
    parser.add_argument(
        "--fixed-paper",
        action="store_true",
        help="verify the first original all-split paper and its pinned assets",
    )
    parser.add_argument(
        "--model-routing",
        action="store_true",
        help="inspect pinned BasicAgent and default local judge SDK routing",
    )
    args = parser.parse_args()
    if args.author_root is None:
        if args.vllm_source is not None or args.fixed_paper or args.model_routing:
            parser.error("requested inspection requires --author-root")
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
    result = inspect_original(args.author_root)
    if args.vllm_source is not None:
        result["vllm_static_browser_dispatch"] = inspect_vllm_browser_dispatch(
            args.vllm_source, result["original_canonical_protocol"]
        )
    if args.fixed_paper:
        result["fixed_original_sample"] = inspect_fixed_paper(args.author_root)
    if args.model_routing:
        result["original_actor_and_judge_routing"] = inspect_author_model_routes(
            args.author_root
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
