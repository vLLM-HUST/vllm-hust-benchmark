"""Original query/all tools using pinned author code; answers never enter input.

Reuse the corpus generation executor. The public author card is obtained separately;
only its two string literals and audited pure builder are loaded, never model code.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import textwrap

from run_openmath2_original import execute


PROFILE = (
    Path(__file__).resolve().parents[1]
    / "configs/workload-native/xlam-author-prompt.json"
)


def author_renderer(card, profile):
    raw = Path(card).read_bytes()
    if (
        len(raw) != profile["bytes"]
        or hashlib.sha256(raw).hexdigest() != profile["sha256"]
    ):
        raise ValueError("pinned original author model card identity mismatch")
    text = raw.decode("utf-8")
    literals = {}
    for name in ("task_instruction", "format_instruction"):
        pattern = (
            r"(?ms)^[ \t]*"
            + name
            + r"\s*=\s*(?P<quote>\"{3}|'{3})(?P<body>.*?)(?P=quote)"
        )
        matches = list(re.finditer(pattern, text))
        if len(matches) != 1:
            raise ValueError("one original author instruction literal required")
        match = matches[0]
        literals[name] = ast.literal_eval(
            match["quote"] + match["body"] + match["quote"]
        )
    lines = text.splitlines()
    starts = [
        i
        for i, line in enumerate(lines)
        if re.match(r"^[ \t]*def build_prompt\(", line)
    ]
    if len(starts) != 1:
        raise ValueError("one original author prompt builder required")
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    selected = [lines[start]]
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        selected.append(line)
    tree = ast.parse(textwrap.dedent("\n".join(selected)))
    function = tree.body[0]
    if len(tree.body) != 1 or not isinstance(function, ast.FunctionDef):
        raise ValueError("pure original author builder required")
    if [arg.arg for arg in function.args.args] != [
        "task_instruction",
        "format_instruction",
        "tools",
        "query",
    ]:
        raise ValueError("original builder signature changed")
    allowed = (
        ast.Module,
        ast.FunctionDef,
        ast.arguments,
        ast.arg,
        ast.Assign,
        ast.AugAssign,
        ast.JoinedStr,
        ast.FormattedValue,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Store,
        ast.Add,
        ast.Return,
        ast.Call,
        ast.Attribute,
    )
    names = {
        "task_instruction",
        "format_instruction",
        "tools",
        "query",
        "prompt",
        "json",
        "xlam_format_tools",
        "str",
        "list",
    }
    for node in ast.walk(tree):
        if (
            not isinstance(node, allowed)
            or isinstance(node, ast.Name)
            and node.id not in names
        ):
            raise ValueError("author builder is not the reviewed pure string function")
        if isinstance(node, ast.Attribute) and ast.unparse(node) != "json.dumps":
            raise ValueError("author builder attribute outside json.dumps")
        if isinstance(node, ast.Call) and (
            ast.unparse(node.func) != "json.dumps"
            or node.keywords
            or len(node.args) != 1
            or ast.unparse(node.args[0]) != "xlam_format_tools"
        ):
            raise ValueError("author builder call outside original tool serialization")
    compiled = compile(tree, "<pinned-author-pure-prompt-builder>", "exec")

    def render(original):
        if (
            set(original) != {"query", "tools"}
            or not isinstance(original["query"], str)
            or not original["query"]
        ):
            raise ValueError("original query and all tools required without answers")
        if not isinstance(original["tools"], str):
            raise ValueError("original JSON-encoded tools string required")
        tools = json.loads(original["tools"])
        if not isinstance(tools, list) or any(
            not isinstance(item, dict) for item in tools
        ):
            raise ValueError("original complete tool list required")
        namespace = {
            "__builtins__": {},
            "json": json,
            "str": str,
            "list": list,
            "xlam_format_tools": tools,
        }
        exec(compiled, namespace)
        return namespace["build_prompt"](
            literals["task_instruction"],
            literals["format_instruction"],
            tools,
            original["query"],
        )

    return render


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "run"], default="plan")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--author-card", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--context", type=int)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--disable-thinking", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--count", type=int, default=3)
    selection.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    profile = json.loads(PROFILE.read_text("utf-8"))
    if args.count < 3:
        parser.error("fixed original prefix requires count>=3")
    if args.mode == "plan":
        print(
            json.dumps(
                {
                    "source_id": "XLAM_TOOL",
                    "status": "PLAN_ONLY",
                    "author_prompt": profile,
                    "original_rows": None if args.full else args.count,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                },
                indent=2,
            )
        )
        return
    if any(
        x is None
        for x in [
            args.data_root,
            args.author_card,
            args.output,
            args.model,
            args.context,
        ]
    ):
        parser.error("run needs external data/card/output and explicit model/context")
    data, output = args.data_root.resolve(strict=True), args.output.resolve()
    if output.is_relative_to(data) or data.is_relative_to(output):
        parser.error("fresh private output must be separate from original data")
    render = author_renderer(args.author_card, profile)
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.assets import verify_file
    from vllm_hust_workload_data.registry import source_spec

    files = [
        x
        for x in source_spec("XLAM_TOOL")["files"]
        if x["role"] in {"raw_source", "data"}
    ]
    if len(files) != 1:
        raise ValueError("one registered original xLAM data file required")
    verify_file(data, files[0])

    def final_verify():
        verify_file(data, files[0])
        author_renderer(args.author_card, profile)

    receipt = execute(
        iter_workload(
            "XLAM_TOOL",
            data_root=data,
            files=[files[0]["path"]],
            limit=None if args.full else args.count,
        ),
        output=output,
        endpoint=args.endpoint,
        model=args.model,
        max_tokens=profile["original_max_new_tokens"],
        context=args.context,
        source_id="XLAM_TOOL",
        expected_count=None if args.full else args.count,
        final_verify=final_verify,
        input_renderer=render,
        protocol_profile="PINNED_AUTHOR_XLAM_QUERY_ALL_TOOLS_CORPUS_GENERATION_NO_OFFICIAL_SCORE",
        chat_template_kwargs={"enable_thinking": False}
        if args.disable_thinking
        else None,
    )
    receipt["author_prompt"] = profile
    receipt["author_model_reproduced"] = args.model == profile["recommended_model"]
    (output / "receipt-private.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "source_id": "XLAM_TOOL",
                "status": receipt["status"],
                "completed_count": receipt["completed_count"],
                "quality_scored": False,
                "source_checkbox_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
