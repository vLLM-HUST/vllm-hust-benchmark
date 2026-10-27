"""Run original BFCL v4 simple_python ranges and the official AST evaluator.

Optional fixed BFCL runtime only. Never executes generated code or real tools.
Use a local NPU endpoint; keep all tasks, references and responses PRIVATE.
"""

import argparse
import copy
from dataclasses import replace
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

REVISION = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"  # pragma: allowlist secret - official commit
ARCHIVE_SHA = "23db73c7b67850696bf4316a92fe97ba54a5dfb46e822bc294e5158b16efffb0"  # pragma: allowlist secret - official archive checksum
MEMBERS = {
    "BFCL_v4_simple_python.json": "82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991",  # pragma: allowlist secret - task checksum
    "possible_answer/BFCL_v4_simple_python.json": "90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc",  # pragma: allowlist secret - scorer asset checksum
}
PREFIX = "ShishirPatil-gorilla-6ea5797/berkeley-function-call-leaderboard/"
EXPECTED_IDS = ["simple_python_0", "simple_python_1", "simple_python_2"]
PROFILE = "local-qwen35-27b-npu-FC"
BASE_PROFILE = "Qwen/Qwen3-32B-FC"


def selected_indices(start=0, count=3, full=False):
    if full:
        if start != 0 or count != 3:
            raise ValueError("full selection cannot be combined with range")
        return list(range(400))
    if start < 0 or count < 1 or start + count > 400:
        raise ValueError("range must be within the original 400 tasks")
    return list(range(start, start + count))


def verify_assets(archive, data_root):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("original BFCL archive identity mismatch")
    raw_files = {}
    with zipfile.ZipFile(archive) as bundle:
        for name, expected in MEMBERS.items():
            member = PREFIX + "bfcl_eval/data/" + name
            if bundle.namelist().count(member) != 1:
                raise ValueError("missing or ambiguous original member")
            raw = (data_root / name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected or raw != bundle.read(
                member
            ):
                raise ValueError(
                    "prepared task/scorer assets differ from publisher bytes"
                )
            raw_files[name] = raw
    rows = [
        json.loads(line)
        for line in raw_files["BFCL_v4_simple_python.json"].decode().splitlines()
    ]
    gold_rows = [
        json.loads(line)
        for line in raw_files["possible_answer/BFCL_v4_simple_python.json"]
        .decode()
        .splitlines()
    ]
    if len(rows) != 400 or len(gold_rows) != 400:
        raise ValueError("original task/scorer count mismatch")
    answers = {row["id"]: row for row in gold_rows}
    if len(answers) != 400 or set(answers) != {row["id"] for row in rows}:
        raise ValueError("ambiguous or misaligned original scorer identities")
    return rows, answers


def verify_runtime_python(archive, package):
    checked = 0
    with zipfile.ZipFile(archive) as bundle:
        for name in bundle.namelist():
            if name.startswith(PREFIX + "bfcl_eval/") and name.endswith(".py"):
                installed = package / name.removeprefix(PREFIX + "bfcl_eval/")
                if installed.read_bytes() != bundle.read(name):
                    raise ValueError("installed official runtime Python differs")
                checked += 1
    if checked != 110:
        raise ValueError("unexpected original runtime file inventory")
    return checked


def original_task_entry(record):
    if record.runtime["kind"] != "task_dataset" or set(record.task["input"]) != {
        "question",
        "function",
    }:
        raise ValueError("requires original question/function task fields")
    return {"id": record.task["task_id"], **copy.deepcopy(record.task["input"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--context-limit", type=int, default=131072)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    try:
        indices = selected_indices(args.start, args.count, args.full)
    except ValueError as error:
        parser.error(str(error))
    url = urlparse(args.endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "localhost", "::1"}
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/") != "/v1"
        or args.context_limit <= 4096
    ):
        parser.error(
            "requires local HTTP /v1 endpoint without credentials and valid context"
        )
    if any(os.environ.get(name) for name in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD")):
        parser.error(
            "clear inherited PYTHONPATH/PYTHONHOME/LD_PRELOAD in the isolated runtime"
        )
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.registry import source_spec

    if source_spec("BFCL")["revision"] != REVISION:
        raise ValueError("install the exact reviewed BFCL data component")
    rows, answers = verify_assets(args.official_archive, args.data_root)
    records = list(
        iter_workload(
            "BFCL",
            data_root=args.data_root,
            files=["BFCL_v4_simple_python.json"],
            limit=indices[-1] + 1,
        )
    )
    entries = [original_task_entry(records[index]) for index in indices]
    if entries != [rows[index] for index in indices]:
        raise ValueError("original selected task fields/identities changed")
    os.environ["REMOTE_OPENAI_BASE_URL"] = args.endpoint.rstrip("/")
    os.environ["REMOTE_OPENAI_API_KEY"] = (
        "EMPTY"  # pragma: allowlist secret - local SDK placeholder, not a credential
    )
    # Optional imports: never load or start a local CPU/CUDA serving model.
    import bfcl_eval
    from bfcl_eval.constants.enums import Language, ReturnFormat
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_ast_entry
    from transformers import AutoTokenizer
    import torch

    if torch.version.cuda is not None or metadata.version("bfcl_eval") != "0.0.0.dev0":
        raise ValueError(
            "requires pinned source build and CPU-only SDK-import Torch runtime"
        )
    verified = verify_runtime_python(
        args.official_archive, Path(bfcl_eval.__file__).parent
    )
    base = MODEL_CONFIG_MAPPING[BASE_PROFILE]
    if base.model_handler is not QwenFCHandler or base.underscore_to_dot is not False:
        raise ValueError("unexpected original Qwen local FC normalization profile")
    MODEL_CONFIG_MAPPING[PROFILE] = replace(
        base,
        model_name=args.model,
        display_name="Qwen3.5-27B local NPU deployment (Qwen FC handler compatibility)",
        url="https://huggingface.co/Qwen/Qwen3.5-27B",
    )
    handler = QwenFCHandler(
        model_name=args.model, temperature=0, registry_name=PROFILE, is_fc_model=True
    )
    handler.model_path_or_id = args.model
    handler.max_context_length = args.context_limit
    handler.tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=False
    )
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "schema_version": "bfcl-simple-python-original/v1",
        "source_id": "BFCL",
        "revision": REVISION,
        "subsuite": "v4/simple_python",
        "status": "STARTED",
        "selected_ids": [entry["id"] for entry in entries],
        "selected_original_indices": indices,
        "full_simple_python_requested": args.full,
        "original_task_rows": 400,
        "original_file_sha256": MEMBERS,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "unchanged_official_python_files": verified,
        "model": args.model,
        "cpu_model_inference": False,
        "generated_code_or_tool_execution": False,
        "scoring_kind": "ORIGINAL_BFCL_AST_ARGUMENT_MATCH_NOT_TOOL_EXECUTION",
        "model_deployment_adaptation": {
            "new_profile": PROFILE,
            "source_compatibility_profile": BASE_PROFILE,
            "original_handler": "QwenFCHandler",
            "underscore_to_dot": False,
            "task_scorer_parser_python_changed": False,
            "paper_model_protocol_equivalence_claim": False,
        },
        "generation_budget": "original OSSHandler min(4096, context-input_tokens-2)",
        "temperature": 0,
        "runtime_versions": {
            name: metadata.version(name)
            for name in (
                "bfcl_eval",
                "torch",
                "transformers",
                "openai",
                "tree_sitter",
                "numpy",
                "soundfile",
            )
        },
        "full_suite_claim": False,
        "performance_claim": False,
        "started_at_unix": time.time(),
        "samples": [],
    }

    def save():
        target = args.output / "receipt.json"
        target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        target.chmod(0o600)

    save()
    try:
        for entry in entries:
            inference = handler._pre_query_processing_prompting(copy.deepcopy(entry))
            inference = handler.add_first_turn_message_prompting(
                inference, copy.deepcopy(entry["question"][0])
            )
            prompt = handler._format_prompt(inference["message"], inference["function"])
            request = Request(
                args.endpoint.rstrip("/").removesuffix("/v1") + "/tokenize",
                data=json.dumps({"model": args.model, "prompt": prompt}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=30) as response:
                tokens = json.load(response)
            count = tokens.get("count", len(tokens.get("tokens", [])))
            if (
                type(count) is not int
                or count <= 0
                or count + 4096 > args.context_limit
            ):
                raise ValueError(
                    "original prompt does not fit; no truncation/replacement"
                )
            item = {
                "id": entry["id"],
                "status": "REQUESTING",
                "preflight_prompt_tokens": count,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "model_request_attempted": True,
            }
            receipt["samples"].append(item)
            save()
            api_response, _ = handler._query_prompting(inference)
            native = handler._parse_query_response_prompting(api_response)
            finish = api_response.choices[0].finish_reason
            if (
                finish not in {"stop", "length"}
                or api_response.usage.prompt_tokens != count
            ):
                raise ValueError(
                    "response termination or full-input token check failed"
                )
            result = _evaluate_single_ast_entry(
                handler,
                entry["id"],
                native["model_responses"],
                answers[entry["id"]]["ground_truth"],
                entry,
                PROFILE,
                "simple_python",
                Language.PYTHON,
                ReturnFormat.PYTHON,
                has_tool_call_tag=True,
            )
            private = args.output / (entry["id"] + ".private.json")
            private.write_text(
                json.dumps(
                    {
                        "api_response": api_response.model_dump(),
                        "official_prompt": prompt,
                        "model_response": native["model_responses"],
                        "official_static_score": result,
                    },
                    default=str,
                ),
                encoding="utf-8",
            )
            private.chmod(0o600)
            item.update(
                status="EXECUTED_AND_OFFICIALLY_STATIC_SCORED",
                finish_reason=finish,
                usage=api_response.usage.model_dump(),
                official_valid=bool(result["valid"]),
                official_error_type=result.get("error_type"),
                private_result_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),
            )
            save()
        receipt.update(
            status="END_TO_END_EXECUTED_AND_SCORED",
            official_valid_count=sum(s["official_valid"] for s in receipt["samples"]),
            finished_at_unix=time.time(),
            source_checkbox_eligible=all(
                s["status"] == "EXECUTED_AND_OFFICIALLY_STATIC_SCORED"
                for s in receipt["samples"]
            ),
        )
    except Exception as error:
        receipt.update(
            status="FAILED_NOT_COMPLETE",
            error_type=type(error).__name__,
            source_checkbox_eligible=False,
        )
        save()
        raise
    save()


if __name__ == "__main__":
    main()
