"""Run original DeepPlanning shopping tasks/tools/scoring via a local NPU API.

Travel is a separate native workflow, not covered by this entrypoint. Default
first-three selection is validation sampling; --full retains all original tasks.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
from urllib.request import urlopen

from prepare_deepplanning_original_assets import prepare, safe_path
from run_docvqa_original import idle, local_endpoint


def select_tasks(rows, full=False):
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)) or not rows:
        raise ValueError("original task IDs missing or duplicated")
    selected = rows if full else rows[:3]
    return [row["id"] for row in selected]


def extract_cases(archive_path, output, level, ids):
    expected = {f"case_{case}" for case in ids}
    found = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if (
                len(parts) < 3
                or parts[0] != f"database_level{level}"
                or parts[1] not in expected
            ):
                continue
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError("linked or special original database member")
            target = safe_path(output, "/".join(parts[1:]))
            target.parent.mkdir(parents=True, exist_ok=True)
            with (
                archive.extractfile(member) as stream,
                target.open("xb") as destination,
            ):
                shutil.copyfileobj(stream, destination)
            found.add(parts[1])
    if found != expected:
        raise ValueError("selected original case databases missing")


def run(args):
    endpoint = local_endpoint(args.endpoint)
    assets = Path(args.assets).resolve()
    output = Path(args.output).resolve()
    if (
        output == assets
        or output.is_relative_to(assets)
        or assets.is_relative_to(output)
    ):
        raise ValueError("output must be independent of original assets")
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("private native outputs must stay outside repository")
    result = {
        "source_id": "DEEPPLANNING",
        "domain": "shopping",
        "status": "PLAN_ONLY",
        "source_checkbox_eligible": False,
        "model": args.model,
        "model_calls": 0,
        "native_max_llm_calls": 400,
        "selection": "all-original-tasks"
        if args.full
        else "first3-original-rows-per-selected-level",
        "levels": args.levels,
        "completed_levels": [],
        "boundary": "Original author tool loops and cart scorer; no travel validation, author-model reproduction, performance or full-source delivery claim.",
    }
    if not args.execute:
        return result
    if output.exists():
        raise ValueError("fresh output required; preserve failed and successful runs")
    prepare(assets, verify_only=True)
    if not idle(endpoint, args.api_server_pid, args.npu_engine_pid):
        raise RuntimeError("research/device priority guard not idle")
    with urlopen(endpoint + "/models", timeout=10) as response:
        models = json.load(response)
    if args.model not in {row["id"] for row in models["data"]}:
        raise ValueError("requested model identity not served")
    output.mkdir(mode=0o700, parents=True)
    code = output / "author"
    shutil.copytree(assets / "source/benchmark/deepplanning", code)
    # Local routing configuration only; native source, task JSON, prompts and
    # model generation parameters remain unchanged.
    (code / "models_config.json").write_text(
        json.dumps(
            {
                "models": {
                    "native-local": {
                        "model_name": args.model,
                        "model_type": "openai",
                        "base_url": endpoint,
                        "api_key_env": "DEEPPLANNING_LOCAL_KEY",  # pragma: allowlist secret - environment variable name, not a credential
                        "temperature": 0.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    os.environ["DEEPPLANNING_LOCAL_KEY"] = (
        "loopback-placeholder"  # pragma: allowlist secret - loopback placeholder, not a credential
    )
    for name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_BASE_URL"):
        os.environ.pop(name, None)
    domain = code / "shoppingplanning"
    sys.path.insert(0, str(domain))
    import openai
    from agent.prompts import prompt_lib
    from agent.shopping_agent import run_agent_inference
    from evaluation.evaluation_pipeline import (
        evaluate_single_case,
        generate_case_report,
        generate_summary_report,
    )

    original_client = openai.OpenAI

    def save():
        (output / "native-public.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )

    def guarded_client(*positional, **kwargs):
        if kwargs.get("base_url") != endpoint:
            raise ValueError("nonlocal model request refused")
        client = original_client(*positional, **kwargs)
        create = client.chat.completions.create

        def transport(**request):
            if not idle(endpoint, args.api_server_pid, args.npu_engine_pid):
                raise RuntimeError("research/device priority guard no longer idle")
            if request.get("model") != args.model:
                raise ValueError("original task routed to unexpected model")
            result["model_calls"] += 1
            save()
            return create(**request)

        client.chat.completions.create = transport
        return client

    openai.OpenAI = guarded_client
    result["status"] = "RUNNING_ORIGINAL_SHOPPING_TASKS"
    save()
    try:
        for level in args.levels:
            query_path = domain / f"data/level_{level}_query_meta.json"
            rows = json.loads(query_path.read_text("utf-8"))
            ids = select_tasks(rows, args.full)
            database = (
                domain / f"database_infered/database_native-local_level{level}_fixed"
            )
            extract_cases(
                assets / f"databases/database_level{level}.tar.gz", database, level, ids
            )
            native = run_agent_inference(
                model="native-local",
                test_data_path=query_path,
                database_dir=database,
                tool_schema_path=domain / "tools/shopping_tool_schema.json",
                system_prompt=getattr(prompt_lib, f"SYSTEM_PROMPT_level{level}"),
                workers=1,
                max_llm_calls=400,
                rerun_ids=ids,
            )
            reports = domain / "result_report" / database.name
            reports.mkdir(parents=True)
            scores = []
            for case in ids:
                score = evaluate_single_case(database / f"case_{case}")
                generate_case_report(score, reports)
                scores.append(score)
            generate_summary_report(scores, reports)
            result["completed_levels"].append(
                {
                    "level": level,
                    "ids": ids,
                    "native_returned_total": native["total"],
                    "native_failed": native["failed"],
                    "score_reports": len(scores),
                }
            )
            save()
            if native["total"] != len(ids) or native["failed"]:
                raise RuntimeError(
                    "original inference returned task failures; saved scoring retained"
                )
        result["status"] = (
            "ORIGINAL_SHOPPING_LOOPS_AND_SCORING_RETURNED_SAVED_AUDIT_PENDING"
        )
    except Exception as error:
        result.update(
            status="FAILED_NOT_COMPLETE",
            error_type=type(error).__name__,
            error=str(error),
        )
        raise
    finally:
        openai.OpenAI = original_client
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-server-pid", type=int, required=True)
    parser.add_argument("--npu-engine-pid", type=int, required=True)
    parser.add_argument(
        "--levels", type=int, nargs="+", choices=(1, 2, 3), default=[1, 2, 3]
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(args.levels) != len(set(args.levels)):
        parser.error("levels must be unique")
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
