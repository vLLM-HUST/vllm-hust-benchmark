"""Run unchanged baselines; never shorten tasks or silently call paid judges.

Default is a plan. Uses the author's original limit, tier and method interfaces.
Source completion and research handoff remain separate from a selected run.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse
from urllib.request import urlopen

from prepare_jsonschemabench_author_source import verify_source
from prepare_longmemeval_v2_original import DATA_PINS, SOURCE_PINS, verify_assets
from prepare_xsum_original_assets import verify as verify_file

READER_PINS = SOURCE_PINS.with_name("longmemeval-v2-reader-pins.json")

LOCAL_SCORERS = {
    "norm_phrase_set_match",
    "norm_phrase_set_match_ordered",
    "mc_choice_match",
    "mc_choice_set_match",
}
JUDGE_SCORERS = {"llm_abstention_checker", "llm_gotchas_checker"}


def prepare_processor_cache(assets, cache, pins):
    assets = Path(assets).resolve(strict=True)
    metadata = [
        item
        for item in pins["model"]["files"]
        if not item["path"].endswith(".safetensors")
    ]
    if not metadata:
        raise ValueError("pinned original processor metadata required")
    for item in metadata:
        path = assets / item["path"]
        if not path.resolve(strict=True).is_relative_to(assets):
            raise ValueError("processor metadata escaped original asset root")
        verify_file(path, item)
    hub = cache / "hub" / ("models--" + pins["model"]["repo_id"].replace("/", "--"))
    (hub / "snapshots").mkdir(parents=True, exist_ok=False)
    (hub / "refs").mkdir()
    (hub / "snapshots" / pins["model"]["revision"]).symlink_to(
        assets, target_is_directory=True
    )
    (hub / "refs/main").write_text(pins["model"]["revision"], encoding="utf-8")
    return len(metadata)


def local_endpoint(value):
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"localhost", "127.0.0.1"}
        or parsed.path.rstrip("/") != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("credential-free loopback HTTP /v1 endpoint required")
    return value.rstrip("/")


def execution_environment(output, *, judge_env=None):
    environment = dict(os.environ, PYTHONNOUSERSITE="1")
    for name in [
        "PYTHONPATH",
        "OPENAI_BASE_URL",
        "LME_DISABLED_EXTERNAL_JUDGE",
        "VLLM_LOCAL_READER_KEY",
        "VLLM_LOCAL_EMBEDDING_KEY",
    ]:
        environment.pop(name, None)
    if judge_env != "OPENAI_API_KEY":
        environment.pop("OPENAI_API_KEY", None)
    environment.update(
        OPENAI_SDK_ONLINE_LEARNING="0",
        OPENAI_AGENTS_DISABLE_TRACING="1",
        HF_HOME=str(output / "hf"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    return environment


def select_original(rows, domain, count=3, full=False, *, start=0):
    if domain not in {"web", "enterprise"} or type(count) is not int or count < 3:
        raise ValueError("original domain and at least three fixed questions required")
    if type(start) is not int or start < 0 or (full and start != 0):
        raise ValueError(
            "fixed original offset must be nonnegative; full cannot use offset"
        )
    selected = [
        (index, row) for index, row in enumerate(rows) if row["domain"] == domain
    ]
    if not full:
        if len(selected) < start + count:
            raise ValueError("original domain has fewer questions than requested")
        selected = selected[start : start + count]
    if not selected or len({row["id"] for _, row in selected}) != len(selected):
        raise ValueError("nonempty unique original question selection required")
    names = [row["eval_function"].split("|", 1)[0].strip() for _, row in selected]
    if set(names) - LOCAL_SCORERS - JUDGE_SCORERS:
        raise ValueError("unknown publisher scorer; no dynamic substitute")
    return selected, sorted(set(names) & JUDGE_SCORERS)


def verify_results(
    native, selected, haystack, metrics, data_root, *, accept_saved_judge_scores=False
):
    records = [
        json.loads(line)
        for line in (native / "per_question.jsonl").read_text("utf-8").splitlines()
    ]
    expected = [row["id"] for _, row in selected]
    if [row["question_id"] for row in records] != expected:
        raise ValueError("original selected question coverage/order differs")
    actual_haystack = json.loads(
        (native / "runtime_inputs/haystack.json").read_text("utf-8")
    )
    if actual_haystack != {qid: haystack[qid] for qid in expected}:
        raise ValueError("original complete ordered haystack relation differs")
    runtime = json.loads((native / "runtime_inputs/questions.json").read_text("utf-8"))
    original_runtime = []
    for _, original in selected:
        item = {key: value for key, value in original.items() if key != "image"}
        if original.get("image") is not None:
            item["question"] = {
                "text": original["question"],
                "image": str((data_root / original["image"]).resolve()),
            }
        original_runtime.append(item)
    if runtime != original_runtime:
        raise ValueError(
            "original runtime question/image/reference materialization differs"
        )
    prompts = [
        json.loads(line)
        for line in (native / "prompt_rows.jsonl").read_text("utf-8").splitlines()
    ]
    if [row["question_id"] for row in prompts] != expected:
        raise ValueError("original prompt coverage/order differs")
    for record, prompt, (_, original) in zip(records, prompts, selected):
        scorer_name = original["eval_function"].split("|", 1)[0].strip()
        if scorer_name not in LOCAL_SCORERS and not (
            accept_saved_judge_scores and scorer_name in JUDGE_SCORERS
        ):
            raise ValueError("external judge re-evaluation is not authorized")
        if record["prompt_messages"] != prompt["prompt_messages"]:
            raise ValueError("saved original prompt relationship differs")
        if (
            any(
                record[key] != original[source]
                for key, source in [
                    ("question_text", "question"),
                    ("answer_gold", "answer"),
                    ("eval_function", "eval_function"),
                ]
            )
            or record["haystack_ids"] != haystack[original["id"]]
        ):
            raise ValueError("original question, reference, scorer or haystack changed")
        usage = record["usage"]
        # The author catches BadRequestError and may exit0 with empty output.
        # Such rows are execution failures, not ordinary wrong answers.
        if (
            not isinstance(record["response_raw"], str)
            or not record["response_raw"]
            or any(
                type(usage[key]) is not int or usage[key] <= 0
                for key in ["prompt_tokens", "completion_tokens"]
            )
        ):
            raise ValueError(
                "empty/zero-usage author fallback is not a completed model request"
            )
        if usage["completion_tokens"] > 20000:
            raise ValueError("original reader completion budget exceeded")
        if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
            raise ValueError("native token accounting differs")
        parsed = metrics.extract_boxed_answer(record["response_raw"])
        if parsed != record["response_parsed_boxed"]:
            raise ValueError("publisher answer parser differs")
        if scorer_name in LOCAL_SCORERS:
            score = metrics.score_to_bool(
                metrics.eval_from_spec(
                    original["eval_function"], parsed, original["answer"]
                )
            )
        else:
            # Keep the original stochastic judgment; never pay/rejudge twice.
            if type(record["score_bool"]) is not bool:
                raise ValueError("original saved judge result must be boolean")
            score = record["score_bool"]
        if metrics.is_unknown(parsed):
            score = False
        if record["score_bool"] != score or record["score"] != float(score):
            raise ValueError(
                "unchanged author score differs; wrong answers must remain"
            )
    aggregated = json.loads((native / "aggregated_metrics.json").read_text("utf-8"))
    if aggregated["overall"]["count_all_questions"] != len(selected):
        raise ValueError("author aggregate coverage differs")
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reader-assets",
        type=Path,
        help="Pinned flat original reader model directory; execute requires this",
    )
    parser.add_argument(
        "--method",
        choices=["no_retrieval", "rag_query_to_slice"],
        default="rag_query_to_slice",
    )
    parser.add_argument("--domain", choices=["web", "enterprise"], required=True)
    parser.add_argument("--tier", choices=["small", "medium"], default="small")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Fixed offset within original domain order; never choose by answer or result",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--reader-endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--embedding-endpoint", default="http://127.0.0.1:19467/v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-paid-judge",
        action="store_true",
        help="Explicit operator opt-in to the original paid evaluator; default blocks it",
    )
    parser.add_argument(
        "--judge-api-key-env",
        default="OPENAI_API_KEY",
        help="Credential environment variable name only, never the credential itself",
    )
    args = parser.parse_args()
    reader = local_endpoint(args.reader_endpoint)
    embedding = local_endpoint(args.embedding_endpoint)
    source = verify_source(args.author_root, SOURCE_PINS)
    data = args.data_root.resolve(strict=True)
    rows = [
        json.loads(line)
        for line in (data / "questions.jsonl").read_text("utf-8").splitlines()
    ]
    selected, judges = select_original(
        rows, args.domain, args.count, args.full, start=args.start
    )
    receipt = {
        "source_id": "LONGMEMEVAL_V2",
        "status": "PLAN_ONLY_NOT_NATIVE_EXECUTED",
        "method": args.method,
        "domain": args.domain,
        "tier": args.tier,
        "original_positions": [index for index, _ in selected],
        "questions": len(selected),
        "required_paid_judge_functions": judges,
        "source_checkbox_eligible": False,
        "raw_payloads_published": False,
    }
    if judges and not args.allow_paid_judge:
        receipt["status"] = (
            "BLOCKED_ORIGINAL_SELECTION_REQUIRES_UNAUTHORIZED_EXTERNAL_JUDGE"
        )
    if not args.execute:
        print(json.dumps(receipt, indent=2))
        return
    if judges and not args.allow_paid_judge:
        raise ValueError(
            "original fixed selection requires external judge; no question replacement or model call"
        )
    if args.reader_assets is None:
        raise ValueError("execute requires pinned original reader assets")
    if judges and not os.environ.get(args.judge_api_key_env):
        raise ValueError(
            "explicit paid-judge opt-in requires a populated credential environment variable; no model call started"
        )
    if args.method != "no_retrieval" and importlib.util.find_spec("torch") is None:
        raise ValueError(
            "original RAG processor requires PyTorch preprocessing; no embedding/model calls started"
        )
    if args.full and args.count != 3:
        raise ValueError("full cannot combine with a custom count")
    for endpoint, model in [(reader, "Qwen/Qwen3.5-9B")] + (
        [(embedding, "Qwen/Qwen3-Embedding-8B")]
        if args.method != "no_retrieval"
        else []
    ):
        with urlopen(endpoint + "/models", timeout=10) as response:
            models = json.load(response)
        if model not in {row["id"] for row in models["data"]}:
            raise ValueError("required original model alias not served; no substitute")
    output = args.output.resolve()
    if any(
        left.is_relative_to(right)
        for left, right in [
            (output, data),
            (data, output),
            (source, output),
            (output, source),
        ]
    ):
        raise ValueError("fresh private output must be separate from originals")
    verify_assets(data, json.loads(DATA_PINS.read_text("utf-8")))
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    native = output / "native"
    command = [
        sys.executable,
        str(source / "evaluation/run_eval.py"),
        "--method",
        args.method,
        "--domain",
        args.domain,
        "--tier",
        args.tier,
        "--data-root",
        str(data),
        "--output-dir",
        str(native),
        "--reader-model",
        "Qwen/Qwen3.5-9B",
        "--reader-base-url",
        reader,
        "--reader-api-key-env",
        "VLLM_LOCAL_READER_KEY",
        "--controller-model",
        "Qwen/Qwen3.5-9B",
        "--controller-base-url",
        reader,
        "--controller-api-key-env",
        "VLLM_LOCAL_READER_KEY",
        "--embedding-model",
        "Qwen/Qwen3-Embedding-8B",
        "--embedding-base-url",
        embedding,
        "--embedding-api-key-env",
        "VLLM_LOCAL_EMBEDDING_KEY",
        "--evaluator-model",
        "gpt-5.2",
        "--evaluator-api-key-env",
        "LME_DISABLED_EXTERNAL_JUDGE",
    ]
    if args.start:
        command += ["--question-ids", *[row["id"] for _, row in selected]]
    elif not args.full:
        command += ["--limit", str(args.count)]
    command += [
        "--reader-temperature",
        "0.6",
        "--reader-top-p",
        "0.95",
        "--reader-top-k",
        "20",
        "--reader-max-concurrent-requests",
        "16",
        "--max-completion-tokens",
        "20000",
        "--memory-context-max-tokens",
        "200000",
        "--reader-enable-thinking",
        "--controller-temperature",
        "0.6",
        "--controller-top-p",
        "0.95",
        "--controller-top-k",
        "20",
        "--prompt-build-max-workers",
        "1",
        "--evaluator-reasoning-effort",
        "medium",
        "--evaluator-max-completion-tokens",
        "4096",
    ]
    environment = execution_environment(
        output, judge_env=args.judge_api_key_env if judges else None
    )
    if judges:
        command[command.index("--evaluator-api-key-env") + 1] = args.judge_api_key_env
    receipt["paid_judge_opt_in"] = bool(judges and args.allow_paid_judge)
    receipt.update(status="RUNNING_NOT_COMPLETE", original_command=command)

    def save():
        (output / "receipt-public.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        receipt["verified_processor_metadata_files"] = prepare_processor_cache(
            args.reader_assets,
            output / "hf",
            json.loads(READER_PINS.read_text("utf-8")),
        )
        save()
        with (output / "original.private.log").open("xb") as log:
            result = subprocess.run(
                command,
                cwd=source,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        receipt["original_exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(
                "unchanged original evaluation failed; preserve output, never replace questions"
            )
        spec = importlib.util.spec_from_file_location(
            "lme_v2_original_metrics", source / "evaluation/qa_eval_metrics.py"
        )
        metrics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(metrics)
        haystack = json.loads(
            (data / f"haystacks/lme_v2_{args.tier}.json").read_text("utf-8")
        )
        receipt["verified_original_results"] = verify_results(
            native,
            selected,
            haystack,
            metrics,
            data,
            accept_saved_judge_scores=bool(judges),
        )
        receipt["local_author_scores_independently_recomputed"] = sum(
            row["eval_function"].split("|", 1)[0].strip() in LOCAL_SCORERS
            for _, row in selected
        )
        receipt["saved_stochastic_judge_rows_retained_not_rejudged"] = sum(
            row["eval_function"].split("|", 1)[0].strip() in JUDGE_SCORERS
            for _, row in selected
        )
        verify_source(source, SOURCE_PINS)
        verify_assets(data, json.loads(DATA_PINS.read_text("utf-8")))
        receipt["status"] = "SELECTED_ORIGINAL_METHOD_AND_LOCAL_AUTHOR_SCORERS_VERIFIED"
        if judges:
            receipt["status"] = (
                "SELECTED_ORIGINAL_METHOD_AND_SAVED_AUTHOR_JUDGE_RESULTS_VALIDATED_NOT_REJUDGED"
            )
    except Exception as exc:
        receipt.update(
            status="FAILED_NOT_COMPLETE", error_type=type(exc).__name__, error=str(exc)
        )
        raise
    finally:
        save()
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
