"""Original DuReader search-demo questions/documents with publisher grading.

Explicit Qwen generative fallback, NOT author Paddle/BiDAF baseline replication.
Every original document paragraph is retained; no oracle evidence selection,
answer leakage, input truncation, failed-question replacement or model retry.
The JSON output instruction is a declared model adapter, not an author prompt.
"""

import argparse
import hashlib
import json
from pathlib import Path

from run_oasst_original_continuation import post, validate_endpoint
from run_toolace_original_continuation import (
    local_tokenizer_files,
    verify_tokenizer_files,
)
from score_dureader_original import load_scorer, score

DATA_SHA = "f7d7174d2cc81738f2554c3aa69f89af7b8d0f4bc808f1c3547469d12469e3a8"  # pragma: allowlist secret - original publisher demo dev checksum
INSTRUCTION = (
    "根据所提供的文档回答原始问题。只输出一个JSON对象，字段为answers、yesno_answers、entity_answers。"
    "answers为只含一个答案字符串的数组；yesno_answers对是非题为只含Yes、No或Depends之一的数组，"
    "非是非题为空数组；entity_answers为只含一个实体字符串数组的数组，非实体题为[[]]。"
    "不要输出Markdown或解释JSON格式。"
)


def originals(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("unchanged original search demo dev required")
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 100 or len({row["question_id"] for row in rows}) != 100:
        raise ValueError("complete original demo identities required")
    return rows


def select(rows, *, start=0, count=3, full=False):
    if full:
        if start != 0 or count != 3:
            raise ValueError("full cannot mix with explicit range")
        return list(enumerate(rows))
    if (
        type(start) is not int
        or type(count) is not int
        or start < 0
        or count < 3
        or start + count > len(rows)
    ):
        raise ValueError("fixed original range of at least three required")
    return list(enumerate(rows))[start : start + count]


def messages(row):
    original = {
        "question": row["question"],
        "documents": [
            {"title": doc["title"], "paragraphs": doc["paragraphs"]}
            for doc in row["documents"]
        ],
    }
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": json.dumps(original, ensure_ascii=False)},
    ]


def reference(row):
    # Missing non-applicable annotation channels use the publisher submission
    # contract's empty placeholders, not fabricated reference answers.
    return {
        "question_type": row["question_type"],
        "answers": row["answers"],
        "yesno_answers": row.get("yesno_answers", []),
        "entity_answers": row.get("entity_answers", [[]]),
        "source": "search",
    }


def prediction(text, row):
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {
        "answers",
        "yesno_answers",
        "entity_answers",
    }:
        raise ValueError("exact native prediction annotation fields required")
    answers, labels, entities = (
        obj["answers"],
        obj["yesno_answers"],
        obj["entity_answers"],
    )
    if (
        not isinstance(answers, list)
        or len(answers) != 1
        or not isinstance(answers[0], str)
    ):
        raise ValueError("one model answer string required")
    if (
        not isinstance(labels, list)
        or len(labels) > 1
        or any(label not in {"Yes", "No", "Depends"} for label in labels)
    ):
        raise ValueError("invalid model yes/no annotation")
    if (
        not isinstance(entities, list)
        or len(entities) != 1
        or not isinstance(entities[0], list)
        or not all(isinstance(entity, str) for entity in entities[0])
    ):
        raise ValueError("invalid model entity annotation")
    return {**obj, "question_type": row["question_type"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--scorer-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    endpoint = validate_endpoint(args.endpoint)
    output, tokenizer_root = args.output.resolve(), args.tokenizer.resolve()
    protected = [
        args.data_file.resolve().parent,
        tokenizer_root,
        args.scorer_root.resolve(),
    ]
    if output.exists() or any(
        output.is_relative_to(root) or root.is_relative_to(output) for root in protected
    ):
        raise ValueError("fresh separate output required")
    rows = originals(args.data_file)
    selected = select(rows, start=args.start, count=args.count, full=args.full)
    module = load_scorer(args.scorer_root)
    from transformers import AutoTokenizer

    identity = local_tokenizer_files(tokenizer_root)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, local_files_only=True, trust_remote_code=False
    )
    prepared = []
    for index, row in selected:
        context = tokenizer.apply_chat_template(
            messages(row),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = len(tokenizer.encode(context, add_special_tokens=False))
        if tokens + 8192 > 131072:
            raise ValueError(
                "complete original input exceeds deployment; never truncate"
            )
        prepared.append((index, row, context, tokens))
    output.mkdir(mode=0o700, parents=True)

    def save(name, obj):
        path = output / name
        path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        path.chmod(0o600)

    receipt = {
        "source_id": "DUREADER",
        "status": "PREPARED",
        "selected_original_dev_positions": [i for i, _ in selected],
        "model": args.model,
        "publisher_baseline_replication": False,
        "model_adapter": "explicit full-original-document generative JSON submission profile",
        "instruction": INSTRUCTION,
        "source_sha256": DATA_SHA,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer_files": identity,
        "max_tokens": 8192,
        "budget_boundary": "explicit generative fallback budget; publisher old extractive baseline has no autoregressive LLM budget",
        "context_capacity": 131072,
        "input_truncated": False,
        "model_calls": 0,
        "cases": [],
        "source_checkbox_eligible": False,
    }
    save("receipt-private.json", receipt)
    save(
        "preflight-private.json",
        [
            {
                "original_index": i,
                "question_id": row["question_id"],
                "question_type": row["question_type"],
                "input_tokens": tokens,
                "reference_answers_empty": not row["answers"],
            }
            for i, row, _, tokens in prepared
        ],
    )
    if args.prepare_only:
        verify_tokenizer_files(tokenizer_root, identity)
        print(
            json.dumps(
                {
                    "status": "PREPARED_NOT_EXECUTED",
                    "selected": len(selected),
                    "input_tokens_min": min(p[3] for p in prepared),
                    "input_tokens_max": max(p[3] for p in prepared),
                    "model_calls": 0,
                }
            )
        )
        return
    predictions, references = {}, {}
    try:
        for index, row, context, tokens in prepared:
            qid = row["question_id"]
            references[qid] = reference(row)
            payload = {
                "model": args.model,
                "prompt": context,
                "max_tokens": 8192,
                "temperature": 0,
                "seed": 20261026,
                "stream": False,
            }
            save(f"case-{index:03d}-request.private.json", payload)
            response = post(endpoint + "/completions", payload)
            save(f"case-{index:03d}-response.private.json", response)
            receipt["model_calls"] += 1
            choice, usage = response["choices"][0], response["usage"]
            if (
                len(response["choices"]) != 1
                or choice["finish_reason"] not in {"stop", "length"}
                or usage["prompt_tokens"] != tokens
                or not 0 <= usage["completion_tokens"] <= 8192
            ):
                raise ValueError("native model terminal integrity failed")
            case = {
                "original_index": index,
                "question_id": qid,
                "finish_reason": choice["finish_reason"],
                "usage": usage,
            }
            try:
                predictions[qid] = prediction(choice["text"], row)
                case["prediction_format"] = "VALID_NATIVE_FIELDS"
            except (ValueError, TypeError) as error:
                # Missing prediction is penalized by the publisher's own rules.
                # Keep raw response, never repair/retry/swap this original task.
                case.update(
                    prediction_format="INVALID_RETAINED_NO_RETRY",
                    error_type=type(error).__name__,
                )
            receipt["cases"].append(case)
            save("predictions-private.json", predictions)
            save("references-private.json", references)
            save("receipt-private.json", receipt)
        verify_tokenizer_files(tokenizer_root, identity)
        originals(args.data_file)
        receipt["scores"] = score(module, predictions, references)
        for name, records in (
            ("native-predictions.private.jsonl", predictions),
            ("native-references.private.jsonl", references),
        ):
            path = output / name
            with path.open("w", encoding="utf-8") as stream:
                for qid, fields in records.items():
                    stream.write(
                        json.dumps({"question_id": qid, **fields}, ensure_ascii=False)
                        + "\n"
                    )
            path.chmod(0o600)
        receipt["status"] = "ORIGINAL_TASKS_EXECUTED_AND_PUBLISHER_METRICS_COMPLETED"
        receipt["invalid_predictions_retained"] = len(references) - len(predictions)
        save("receipt-private.json", receipt)
        print(
            json.dumps(
                {
                    key: receipt[key]
                    for key in (
                        "status",
                        "model_calls",
                        "invalid_predictions_retained",
                        "scores",
                    )
                }
            )
        )
    except Exception as error:
        receipt.update(status="FAILED_NOT_REPLACED", error_type=type(error).__name__)
        save("receipt-private.json", receipt)
        raise


if __name__ == "__main__":
    main()
