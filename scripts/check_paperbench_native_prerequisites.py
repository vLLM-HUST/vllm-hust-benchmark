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
    return {
        "status": "FIXED_ORIGINAL_FULL_SPLIT_PAPER_ASSETS_VERIFIED_NOT_RUN",
        "publisher_split": "all",
        "publisher_split_papers": full_count,
        "sample_paper_id": selected,
        "selection_rule": "first nonempty entry in the pinned original all.txt",
        "verified_original_paper_files": len(files),
        "verified_original_paper_bytes": sum(item["bytes"] for item in files),
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
    args = parser.parse_args()
    if args.author_root is None:
        if args.vllm_source is not None or args.fixed_paper:
            parser.error("--vllm-source and --fixed-paper require --author-root")
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
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
