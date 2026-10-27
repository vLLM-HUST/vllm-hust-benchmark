"""Pinned official StructEval JSON tasks, original workflow and scorer, local NPU API."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import zipfile


def select_original_rows(
    rows, input_type="Text", output_type="JSON", *, start=0, count=3, full=False
):
    available = [
        (index, row)
        for index, row in enumerate(rows)
        if row["input_type"] == input_type
        and row["output_type"] == output_type
        and not row["rendering"]
    ]
    if not available:
        raise ValueError("no original non-rendering tasks for selected protocol")
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
        "--input-type",
        default="Text",
        choices=["Text", "CSV", "JSON", "XML", "YAML", "TOML"],
    )
    parser.add_argument(
        "--output-type", default="JSON", choices=["CSV", "JSON", "XML", "YAML", "TOML"]
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--all-selected",
        action="store_true",
        help="All original rows of the explicit non-rendering protocol",
    )
    parser.add_argument(
        "--all-json",
        action="store_true",
        help="All original Text-to-JSON tasks only; excludes format-to-JSON conversions",
    )
    args = parser.parse_args()
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
    )
    dataset = args.output / "original-selected.private.json"
    dataset.write_text(json.dumps([r for _, r in selected]), encoding="utf-8")
    model = "openai/" + args.model
    original_acompletion = litellm.acompletion

    async def local_only(*positional, **kwargs):
        if kwargs.get("model") != model or kwargs.get("api_base") != args.endpoint:
            raise RuntimeError("unapproved model or external judge refused")
        return await original_acompletion(*positional, **kwargs)

    litellm.acompletion = local_only
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
            "rendering": False,
        },
        "available_selected_protocol_tasks": available,
        "selection": "explicit non-rendering protocol and original order/range, fixed before model calls",
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": None,
        "prompt": "official build_prompt(query) including original official code markers",
        "scorer": "official syntax and path validation, no vision/LLM judge",
        "status": "STARTED",
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
                judge_model=None,
                concurrency=1,
                render_concurrency=1,
                temperature=0.0,
                max_tokens=None,
                timeout=300,
                max_retries=0,
                api_base=args.endpoint,
                litellm_params={
                    "api_key": "local-unused"  # pragma: allowlist secret - dummy loopback key
                },  # pragma: allowlist secret - dummy loopback key
                split="full",
            )
        )
        scored = json.loads(Path(result["evaluation"]).read_text())
        if [r["task_id"] for r in scored] != receipt["selected_samples"]:
            raise ValueError("scored task IDs differ from preselected original rows")
        receipt.update(
            status="END_TO_END_EXECUTED_AND_SCORED",
            unique_tasks=len(scored),
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
