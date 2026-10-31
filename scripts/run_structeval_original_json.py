"""Pinned StructEval original tasks/scorer with explicit isolated render dispatch."""

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import zipfile


def no_implicit_sdk_retries(kwargs):
    """Honor the selected zero-retry run at both library and provider layers."""
    for key in ("max_retries", "num_retries"):
        if kwargs.get(key) not in (None, 0):
            raise ValueError("conflicting native retry configuration; do not overwrite")
    return {**kwargs, "max_retries": 0, "num_retries": 0}


def positive_timeout(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("finite positive request timeout required")
    return result


def select_original_rows(
    rows,
    input_type="Text",
    output_type="JSON",
    *,
    start=0,
    count=3,
    full=False,
    rendering=False,
):
    available = [
        (index, row)
        for index, row in enumerate(rows)
        if row["input_type"] == input_type
        and row["output_type"] == output_type
        and row["rendering"] == rendering
    ]
    if not available:
        raise ValueError("no original tasks for selected protocol")
    if full:
        if start != 0 or count != 3:
            raise ValueError("full selection cannot be combined with range")
        return available, len(available)
    if start < 0 or count < 1 or start + count > len(available):
        raise ValueError("range must be within the original protocol")
    return available[start : start + count], len(available)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument(
        "--request-timeout",
        type=positive_timeout,
        default=300,
        help="HTTP transport timeout in seconds; does not change original output or step budgets",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="Explicit author client retries; hidden SDK retries remain disabled",
    )
    parser.add_argument(
        "--input-type",
        default="Text",
        choices=[
            "Text",
            "CSV",
            "JSON",
            "XML",
            "YAML",
            "TOML",
            "HTML",
            "Markdown",
            "React",
            "Vue",
            "Matplotlib",
        ],
    )
    parser.add_argument(
        "--output-type",
        default="JSON",
        choices=[
            "CSV",
            "JSON",
            "XML",
            "YAML",
            "TOML",
            "Matplotlib",
            "Typst",
            "SVG",
            "HTML",
            "Canvas",
            "Markdown",
            "Latex",
            "Tikz",
            "TikZ",
        ],
    )
    parser.add_argument("--render-queue", type=Path)
    parser.add_argument("--render-image")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--all-selected",
        action="store_true",
        help="All original rows of the explicit protocol",
    )
    parser.add_argument(
        "--all-json",
        action="store_true",
        help="All original Text-to-JSON tasks only; excludes format-to-JSON conversions",
    )
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error("nonnegative explicit client retry count required")
    rendering = args.output_type in {
        "Matplotlib",
        "Typst",
        "SVG",
        "HTML",
        "Canvas",
        "Markdown",
        "Latex",
        "Tikz",
        "TikZ",
    }
    if rendering != bool(args.render_queue and args.render_image):
        parser.error(
            f"{args.output_type} requires both private render queue and pinned image"
        )
    if not rendering and (args.render_queue or args.render_image):
        parser.error("render dispatch requires an explicitly supported render protocol")
    if args.all_json and (args.input_type != "Text" or args.output_type != "JSON"):
        parser.error("--all-json is only the historical Text-to-JSON selection")
    url = urlparse(args.endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost"}
        or url.username
        or url.password
    ):
        parser.error("only local HTTP NPU model endpoints permitted")
    if (
        hashlib.sha256(args.data_file.read_bytes()).hexdigest()
        != "66704021b99dd10b242e7a74c44f6a222c2fa6779edbb0476ec8cf5e18132fd8"  # pragma: allowlist secret - original data checksum
    ):  # pragma: allowlist secret - original data checksum
        raise ValueError("original dataset identity mismatch")
    if (
        hashlib.sha256(args.official_archive.read_bytes()).hexdigest()
        != "43593f97711c77c15c4e3711fe94bea188c58e718535893c0640fa17095360bb"  # pragma: allowlist secret - official source checksum
    ):  # pragma: allowlist secret - official source checksum
        raise ValueError("official source archive identity mismatch")
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    os.environ.update(LITELLM_LOCAL_MODEL_COST_MAP="True", LITELLM_TELEMETRY="False")
    import pyarrow.parquet as pq
    import litellm
    import structeval.workflows as workflow

    package_root = Path(workflow.__file__).parent
    with zipfile.ZipFile(args.official_archive) as archive:
        for name in archive.namelist():
            if "/structeval/" in name and name.endswith(".py"):
                relative = name.split("/structeval/", 1)[1]
                if (package_root / relative).read_bytes() != archive.read(name):
                    raise ValueError("installed official workflow source mismatch")
    rows = pq.read_table(args.data_file).to_pylist()
    selected, available = select_original_rows(
        rows,
        args.input_type,
        args.output_type,
        start=args.start,
        count=args.count,
        full=args.all_json or args.all_selected,
        rendering=rendering,
    )
    dataset = args.output / "original-selected.private.json"
    dataset.write_text(json.dumps([r for _, r in selected]), encoding="utf-8")
    model = "openai/" + args.model
    original_acompletion = litellm.acompletion
    audit_root = args.output / "model-transport-private"
    audit_root.mkdir(mode=0o700)
    call_number = 0

    async def local_only(*positional, **kwargs):
        nonlocal call_number
        if kwargs.get("model") != model or kwargs.get("api_base") != args.endpoint:
            raise RuntimeError("unapproved model or external judge refused")
        kwargs = no_implicit_sdk_retries(kwargs)
        if args.disable_thinking:
            extra = kwargs.setdefault("extra_body", {})
            if "chat_template_kwargs" in extra:
                raise ValueError(
                    "do not overwrite existing native template configuration"
                )
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        call_number += 1
        call_id = str(call_number).zfill(6)
        (audit_root / (call_id + ".request.json")).write_text(
            json.dumps(
                {"positional": positional, "kwargs": kwargs}, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        try:
            response = await original_acompletion(*positional, **kwargs)
        except asyncio.CancelledError:
            (audit_root / (call_id + ".error.json")).write_text(
                json.dumps(
                    {
                        "error_type": "CancelledError",
                        "completion_received": False,
                        "model_outcome": "UNKNOWN_NOT_A_TASK_SCORE",
                    }
                ),
                encoding="utf-8",
            )
            raise
        except Exception as error:
            (audit_root / (call_id + ".error.json")).write_text(
                json.dumps({"error_type": type(error).__name__}), encoding="utf-8"
            )
            raise
        (audit_root / (call_id + ".response.json")).write_text(
            response.model_dump_json(), encoding="utf-8"
        )
        return response

    litellm.acompletion = local_only
    queue = None
    if rendering:
        import structeval.render_engine.main as render_main
        from structeval_render_queue import (
            CanvasRenderQueue,
            HtmlRenderQueue,
            LatexRenderQueue,
            MarkdownRenderQueue,
            MatplotlibRenderQueue,
            SvgRenderQueue,
            TypstRenderQueue,
        )

        queue_class = {
            "Typst": TypstRenderQueue,
            "Matplotlib": MatplotlibRenderQueue,
            "SVG": SvgRenderQueue,
            "HTML": HtmlRenderQueue,
            "Canvas": CanvasRenderQueue,
            "Markdown": MarkdownRenderQueue,
            "Latex": LatexRenderQueue,
            "Tikz": LatexRenderQueue,
            "TikZ": LatexRenderQueue,
        }[args.output_type]
        queue = queue_class(
            args.render_queue,
            image_root=args.output / "official-workflow" / "images",
            image=args.render_image,
            timeout=900,
        )
        if args.output_type in {"SVG", "HTML", "Canvas", "Markdown"}:

            async def remote_browser(task_id, parsed_code, img_output_path):
                return await asyncio.to_thread(
                    queue.render, task_id, parsed_code, img_output_path
                )

            render_main.ASYNC_RENDERERS[args.output_type.lower()] = remote_browser
        elif args.output_type in {"Latex", "Tikz", "TikZ"}:
            render_main.render_latex_to_png = queue.render
        else:
            render_main.THREAD_RENDERERS[args.output_type.lower()] = queue.render
        original_render_file = workflow.render_file_async

        async def guarded_render(**kwargs):
            result = await original_render_file(**kwargs)
            queue.require_ready_for_judge()
            return result

        workflow.render_file_async = guarded_render
    receipt = {
        "source_id": "STRUCTEVAL",
        "source_revision": "936be10548688c6d417cd435dd906158f19ba14b",  # pragma: allowlist secret - public dataset revision
        "runner_revision": "7781339a5b033965ac664133543ebd10773643dd",  # pragma: allowlist secret - public official branch revision
        "selected_samples": [r["task_id"] for _, r in selected],
        "original_row_indices": [i for i, _ in selected],
        "available_text_to_json_tasks": available,
        "selected_protocol": {
            "input_type": args.input_type,
            "output_type": args.output_type,
            "rendering": rendering,
        },
        "available_selected_protocol_tasks": available,
        "selection": "explicit protocol and original order/range, fixed before model calls",
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": None,
        "request_timeout_seconds": args.request_timeout,
        "outer_max_retries": args.max_retries,
        "sdk_max_retries": 0,
        "prompt": "official build_prompt(query) including original official code markers",
        "scorer": "official complete VQA and final evaluation"
        if rendering
        else "official syntax and path validation, no vision/LLM judge",
        "status": "STARTED",
        "thinking_explicitly_disabled": args.disable_thinking,
        "render_dispatch_only_original_parser_and_vqa_unchanged": rendering,
        "started_at_unix": time.time(),
        "all_text_to_json_requested": args.all_json,
    }
    target = args.output / "receipt.json"
    target.write_text(json.dumps(receipt, indent=2))
    try:
        result = asyncio.run(
            workflow.run_workflow_async(
                dataset=dataset,
                output_dir=args.output / "official-workflow",
                model=model,
                judge_model=model if rendering else None,
                concurrency=1,
                render_concurrency=1,
                temperature=0.0,
                max_tokens=None,
                timeout=args.request_timeout,
                max_retries=args.max_retries,
                api_base=args.endpoint,
                litellm_params={
                    "api_key": "local-unused"  # pragma: allowlist secret - dummy loopback key
                },  # pragma: allowlist secret - dummy loopback key
                split="full",
                judge_api_base=args.endpoint if rendering else None,
                judge_litellm_params={
                    "api_key": "local-unused"  # pragma: allowlist secret - dummy loopback key
                }
                if rendering
                else None,  # pragma: allowlist secret - dummy loopback key
            )
        )
        scored = json.loads(Path(result["evaluation"]).read_text())
        if [r["task_id"] for r in scored] != receipt["selected_samples"]:
            raise ValueError("scored task IDs differ from preselected original rows")
        receipt.update(
            status="END_TO_END_EXECUTED_AND_SCORED",
            unique_tasks=len(scored),
            model_transport_calls=call_number,
            render_dispatch_receipts=queue.receipts if queue else [],
            tasks=[
                {
                    k: r.get(k)
                    for k in (
                        "task_id",
                        "render_score",
                        "key_validation_score",
                        "final_eval_score",
                    )
                }
                for r in scored
            ],
        )
    except BaseException as error:
        receipt.update(status="EXECUTION_ERROR", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        target.write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
