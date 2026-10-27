"""Original DeepPlanning travel Agent, report conversion and constraint scoring.

Both Agent and report-conversion model route to an explicitly declared local
fallback. This is not reproduction of the author's qwen-plus conversion score.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from urllib.request import urlopen
import zipfile

from prepare_deepplanning_original_assets import prepare, safe_path
from run_deepplanning_original_shopping import select_tasks
from run_docvqa_original import idle, local_endpoint


def extract_cases(archive_path, output, language, ids):
    expected = {f"id_{case}" for case in ids}
    found = set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            parts = PurePosixPath(member.filename).parts
            if (
                len(parts) < 3
                or parts[0] != f"database_{language}"
                or parts[1] not in expected
            ):
                continue
            if member.is_dir():
                continue
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("linked original database member")
            target = safe_path(output, "/".join(parts[1:]))
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as stream, target.open("xb") as destination:
                shutil.copyfileobj(stream, destination)
            found.add(parts[1])
    if found != expected:
        raise ValueError("selected original travel databases missing")


def run(args):
    endpoint = local_endpoint(args.endpoint)
    assets = Path(args.assets).resolve()
    output = Path(args.output).resolve()
    if output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("independent output required")
    if output.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("private native output must stay outside repository")
    result = {
        "source_id": "DEEPPLANNING",
        "domain": "travel",
        "language": args.language,
        "status": "PLAN_ONLY",
        "source_checkbox_eligible": False,
        "model": args.model,
        "conversion_model": args.model,
        "model_calls": 0,
        "native_max_llm_calls": 400,
        "full": args.full,
        "boundary": "Original Agent, tool use, conversion prompt and constraint scorer; local model fallback also replaces author qwen-plus report converter, so not an author-model score reproduction.",
    }
    if not args.execute:
        return result
    if output.exists():
        raise ValueError("fresh output required; preserve prior results")
    prepare(assets, verify_only=True)
    if not idle(endpoint, args.api_server_pid, args.npu_engine_pid):
        raise RuntimeError("research/device priority guard not idle")
    with urlopen(endpoint + "/models", timeout=10) as response:
        models = json.load(response)
    if args.model not in {row["id"] for row in models["data"]}:
        raise ValueError("declared model not served")
    output.mkdir(parents=True, mode=0o700)
    code = output / "author"
    shutil.copytree(assets / "source/benchmark/deepplanning", code)
    local = {
        "model_name": args.model,
        "model_type": "openai",
        "base_url": endpoint,
        "api_key_env": "DEEPPLANNING_LOCAL_KEY",  # pragma: allowlist secret - environment variable name, not a credential
        "temperature": 0.0,
    }
    (code / "models_config.json").write_text(
        json.dumps({"models": {"native-local": local, "qwen-plus": local}}),
        encoding="utf-8",
    )
    os.environ["DEEPPLANNING_LOCAL_KEY"] = (
        "loopback-placeholder"  # pragma: allowlist secret - loopback placeholder, not a credential
    )
    for name in ("OPENAI_API_KEY", "DASHSCOPE_API_KEY", "OPENAI_BASE_URL"):
        os.environ.pop(name, None)
    domain = code / "travelplanning"
    queries = domain / f"data/travelplanning_query_{args.language}.json"
    ids = select_tasks(json.loads(queries.read_text("utf-8")), args.full)
    result["ids"] = ids
    database = domain / "database" / f"database_{args.language}"
    extract_cases(
        assets / f"databases/database_{args.language}.zip", database, args.language, ids
    )
    sys.path.insert(0, str(domain))
    import openai
    from agent.tools_fn_agent import run_agent_inference
    from evaluation.convert_report import convert_reports
    from evaluation.eval_converted import evaluate_plans

    native_output = output / "native-results"
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
            if request.get("model") != args.model:
                raise ValueError("unexpected model route")
            if not idle(endpoint, args.api_server_pid, args.npu_engine_pid):
                raise RuntimeError("research/device priority guard no longer idle")
            result["model_calls"] += 1
            save()
            return create(**request)

        client.chat.completions.create = transport
        return client

    openai.OpenAI = guarded_client
    result["status"] = "RUNNING_ORIGINAL_TRAVEL_TASKS"
    save()
    try:
        native = run_agent_inference(
            model="native-local",
            language=args.language,
            test_data_path=queries,
            database_dir=database,
            tool_schema_path=domain / f"tools/tool_schema_{args.language}.json",
            output_dir=native_output,
            workers=1,
            max_llm_calls=400,
            rerun_ids=ids,
        )
        result["inference"] = {
            key: native[key] for key in ("total", "success", "failed")
        }
        save()
        conversion = convert_reports(
            result_dir=native_output,
            language=args.language,
            workers=1,
            skip_existing=False,
            verbose=True,
        )
        result["conversion"] = {
            key: value for key, value in conversion.items() if key != "results"
        }
        save()
        scores = evaluate_plans(
            result_dir=native_output,
            test_data_path=queries,
            database_dir=database,
            verbose=True,
        )
        result["scoring"] = {
            key: value for key, value in scores.items() if key != "results"
        }
        if (
            native["total"] != len(ids)
            or native["failed"]
            or conversion.get("failed", 0)
        ):
            raise RuntimeError("native failures retained; not complete")
        result["status"] = (
            "ORIGINAL_TRAVEL_LOOPS_CONVERSION_AND_SCORING_RETURNED_SAVED_AUDIT_PENDING"
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
    for option in ("assets", "output", "endpoint", "model"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--api-server-pid", type=int, required=True)
    parser.add_argument("--npu-engine-pid", type=int, required=True)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
