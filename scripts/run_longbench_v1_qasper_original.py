"""Run original LongBench qasper rows with unchanged QA-F1 scoring.

Install separately: numpy==2.2.6 jieba==0.42.1 rouge==1.0.1 fuzzywuzzy==0.18.0
six==1.17.0, plus the exact reviewed workload-data component. Output is PRIVATE.
No question truncation or budget fallback; model-native chat transport is explicit.
"""

import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile

SOURCE_SHA = "2f92f9953e3dd2fd3915a132a0a4cb53acb879af6ef1e98962129a6f3fe262bb"  # pragma: allowlist secret - official source archive checksum
DATA_SHA = "29aa07d2a63f36f4fb8e8cd200a3428ee3126d750bde6af1c8f9bc41c2366854"  # pragma: allowlist secret - original qasper checksum
SOURCE_REVISION = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"  # pragma: allowlist secret - public source revision
DATA_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"  # pragma: allowlist secret - public dataset revision
EXPECTED_IDS = [
    "3ac3eef636db11635a21a61804cb28e92c546a5686dd1e12",  # pragma: allowlist secret - public task ID
    "db6c4d611b78c840972b57ee0b245003566b982feaff1084",  # pragma: allowlist secret - public task ID
    "e419f2bff9d2ab7c3b60b3250caccd2d9ae1285ec3e8e818",  # pragma: allowlist secret - public task ID
]
DEPENDENCIES = {
    "numpy": "2.2.6",
    "jieba": "0.42.1",
    "rouge": "1.0.1",
    "fuzzywuzzy": "0.18.0",
    "six": "1.17.0",
}


def render_prompt(record, template):
    if record.runtime["kind"] != "task_dataset" or set(record.task["input"]) != {
        "context",
        "input",
    }:
        raise ValueError("requires original document/question fields only")
    return template.format(**record.task["input"])


def official_resources(archive):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA:
        raise ValueError("official source archive identity mismatch")
    with zipfile.ZipFile(archive) as bundle:
        prefix = "THUDM-LongBench-2e00731/LongBench/"
        names = [
            "eval.py",
            "metrics.py",
            "config/dataset2prompt.json",
            "config/dataset2maxlen.json",
        ]
        if any(bundle.namelist().count(prefix + name) != 1 for name in names):
            raise ValueError("missing or ambiguous official source member")
        return {name: bundle.read(prefix + name) for name in names}


def select_original_records(records, *, start=0, count=3, full=False):
    """Select original positions, never filter on task length or model outcomes."""
    if full:
        if start != 0 or count != 3:
            raise ValueError("full selection cannot be combined with a range")
        return list(enumerate(records))
    if start < 0 or count < 1 or start + count > len(records):
        raise ValueError("range must stay within the original dataset")
    return list(enumerate(records))[start : start + count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--official-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--context-limit", type=int, default=131072)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--full", action="store_true", help="All 200 original Qasper rows"
    )
    args = parser.parse_args()
    if args.full and (args.start != 0 or args.count != 3):
        parser.error("--full cannot be combined with a range")
    route = urlparse(args.endpoint)
    if (
        route.scheme != "http"
        or route.hostname not in {"127.0.0.1", "localhost", "::1"}
        or route.username
        or route.password
        or route.query
        or route.fragment
        or route.path.rstrip("/") != "/v1"
        or args.context_limit <= 128
    ):
        parser.error(
            "requires local HTTP /v1 endpoint without credentials and valid context limit"
        )
    from vllm_hust_workload_data import iter_workload
    from vllm_hust_workload_data.registry import source_spec

    source = source_spec("LONGBENCH")
    if source["revision"] != DATA_REVISION or not any(
        f["path"] == "qasper.jsonl" and f["sha256"] == DATA_SHA for f in source["files"]
    ):
        raise ValueError("install the exact reviewed LongBench data component")
    versions = {name: metadata.version(name) for name in DEPENDENCIES}
    if versions != DEPENDENCIES:
        raise ValueError("requires reviewed isolated scorer dependency versions")
    if (
        hashlib.sha256((args.data_root / "qasper.jsonl").read_bytes()).hexdigest()
        != DATA_SHA
    ):
        raise ValueError("original qasper file identity mismatch")
    all_records = list(
        iter_workload("LONGBENCH", data_root=args.data_root, files=["qasper.jsonl"])
    )
    if (
        len(all_records) != 200
        or [r.task["task_id"] for r in all_records[:3]] != EXPECTED_IDS
    ):
        raise ValueError("requires the original 200-row qasper order and identity")
    selected = select_original_records(
        all_records, start=args.start, count=args.count, full=args.full
    )
    records = [record for _, record in selected]
    resources = official_resources(args.official_archive)
    template = json.loads(resources["config/dataset2prompt.json"])["qasper"]
    budget = json.loads(resources["config/dataset2maxlen.json"])["qasper"]
    if budget != 128:
        raise ValueError("unexpected official qasper generation budget")
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    official = args.output / "official-source"
    official.mkdir(mode=0o700)
    for name in ("eval.py", "metrics.py"):
        (official / name).write_bytes(resources[name])
    sys.path.insert(0, str(official))
    spec = importlib.util.spec_from_file_location(
        "original_longbench_eval", official / "eval.py"
    )
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    receipt = {
        "schema_version": "longbench-qasper-original/v1",
        "source_id": "LONGBENCH",
        "subsuite": "qasper",
        "source_revision": SOURCE_REVISION,
        "data_revision": DATA_REVISION,
        "data_sha256": DATA_SHA,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_source_file_sha256": {
            k: hashlib.sha256(v).hexdigest() for k, v in resources.items()
        },
        "selected_ids": [r.task["task_id"] for r in records],
        "original_row_indices": [index for index, _ in selected],
        "selection": "ORIGINAL_ORDER_RANGE_OR_FULL_FIXED_BEFORE_EXECUTION",
        "all_original_qasper_requested": args.full,
        "original_qasper_rows": 200,
        "model": args.model,
        "scorer_dependencies": versions,
        "transport": "ORIGINAL_TEMPLATE_VERBATIM_AS_SINGLE_USER_MESSAGE_WITH_MODEL_NATIVE_CHAT_TEMPLATE_NO_TRUNCATION",
        "generation_config": {
            "temperature": 0,
            "max_tokens": budget,
            "n": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "declared_context_limit": args.context_limit,
        "started_at_unix": time.time(),
        "status": "STARTED",
        "full_suite_claim": False,
        "performance_claim": False,
        "samples": [],
    }

    def save():
        target = args.output / "receipt.json"
        target.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        target.chmod(0o600)

    def post(url, payload):
        with urlopen(
            Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=600,
        ) as response:
            return json.load(response)

    save()
    predictions = []
    try:
        for index, record in selected:
            prompt = render_prompt(record, template)
            messages = [{"role": "user", "content": prompt}]
            item = {
                "row_index": index,
                "id": record.task["task_id"],
                "status": "PREFLIGHT",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "model_request_attempted": False,
            }
            receipt["samples"].append(item)
            save()
            tokens = post(
                args.endpoint.rstrip("/").removesuffix("/v1") + "/tokenize",
                {
                    "model": args.model,
                    "messages": messages,
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            count = tokens.get("count", len(tokens.get("tokens", [])))
            item["preflight_prompt_tokens"] = count
            if (
                type(count) is not int
                or count <= 0
                or count + budget > args.context_limit
            ):
                item["status"] = "ORIGINAL_INPUT_EXCEEDS_CONTEXT_NOT_TRUNCATED"
                raise ValueError(
                    "original prompt does not fit declared context; no replacement task"
                )
            payload = {
                "model": args.model,
                "messages": messages,
                **receipt["generation_config"],
            }
            item.update(status="REQUESTING", model_request_attempted=True)
            save()
            result = post(args.endpoint.rstrip("/") + "/chat/completions", payload)
            choices = result.get("choices", [])
            if len(choices) != 1 or choices[0].get("finish_reason") not in {
                "stop",
                "length",
            }:
                raise ValueError("invalid terminal response")
            prediction = choices[0].get("message", {}).get("content")
            if (
                not isinstance(prediction, str)
                or not prediction
                or result["usage"]["prompt_tokens"] != count
            ):
                raise ValueError("missing completion or input token roundtrip mismatch")
            outcome = record.outcome_record()
            private = args.output / f"row-{index}.private.json"
            private.write_text(
                json.dumps(
                    {
                        "request": payload,
                        "response": result,
                        "original_answers": outcome["answers"],
                    }
                ),
                encoding="utf-8",
            )
            private.chmod(0o600)
            score = evaluator.scorer(
                "qasper", [prediction], [outcome["answers"]], outcome["all_classes"]
            )
            predictions.append(prediction)
            item.update(
                status="EXECUTED_AND_OFFICIALLY_SCORED",
                finish_reason=choices[0]["finish_reason"],
                usage=result["usage"],
                official_f1_percent=score,
                private_result_sha256=hashlib.sha256(private.read_bytes()).hexdigest(),
            )
            save()
        receipt.update(
            status="END_TO_END_EXECUTED_AND_SCORED",
            official_f1_percent=evaluator.scorer(
                "qasper",
                predictions,
                [r.outcome_record()["answers"] for r in records],
                records[0].outcome_record()["all_classes"],
            ),
            finished_at_unix=time.time(),
            source_checkbox_eligible=len(records) >= 3
            and all(
                s["status"] == "EXECUTED_AND_OFFICIALLY_SCORED"
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
