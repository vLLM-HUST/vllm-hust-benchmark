"""Original QASPER v0.3 question/answer/evidence task and publisher scoring.

Full original paper text and figure/table metadata, no oracle evidence or input
truncation. Declared generative text/caption model fallback, not the author's
AllenNLP/LED model/training replication or image-pixel inference. No retry or
prediction repair; malformed native submissions remain missing and score zero.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from urllib.request import Request, urlopen

from run_oasst_original_continuation import validate_endpoint
from run_toolace_original_continuation import (
    local_tokenizer_files,
    verify_tokenizer_files,
)

FILES = {
    "dev": (
        "qasper-dev-v0.3.json",
        1005,
        "2ae7ee62a65b1c4225791c70de80c2aad4e8998cf1fd4f09a53103db4f21af93",  # pragma: allowlist secret - public original dev identity
    ),
    "test": (
        "qasper-test-v0.3.json",
        1451,
        "6e29ad410e6e39aa1936017fb965b30a20eb2e7751997f55b97c9d281aa884e5",  # pragma: allowlist secret - public original test identity
    ),
}
EVALUATOR_SHA = "781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe"  # pragma: allowlist secret - original publisher evaluator identity
INSTRUCTION = (
    "Answer the original question using the supplied complete research paper. "
    "Return only a JSON object with predicted_answer (a string) and predicted_evidence "
    "(an array of evidence strings). Use Yes, No, or Unanswerable when appropriate. "
    "Otherwise give a concise answer. Quote complete original paragraphs as textual "
    "evidence, preserving exact wording. Do not output Markdown formatting."
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def originals(root, split):
    name, count, expected = FILES[split]
    path = Path(root) / name
    if sha(path) != expected:
        raise ValueError("unchanged original QASPER v0.3 split required")
    papers = json.loads(path.read_text("utf-8"))
    rows = [
        (paper_id, index, question)
        for paper_id, paper in papers.items()
        for index, question in enumerate(paper["qas"])
    ]
    if len(rows) != count or len({q["question_id"] for _, _, q in rows}) != count:
        raise ValueError("complete original question identities required")
    return papers, rows


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
        raise ValueError("preselect original range of at least three")
    return list(enumerate(rows))[start : start + count]


def messages(paper, question):
    context = {
        key: paper[key]
        for key in ("title", "abstract", "full_text", "figures_and_tables")
    }
    context["question"] = question["question"]
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]


def subset_gold(papers, selected):
    result = {}
    for _, (paper_id, _, question) in selected:
        if paper_id not in result:
            result[paper_id] = {
                key: deepcopy(value)
                for key, value in papers[paper_id].items()
                if key != "qas"
            }
            result[paper_id]["qas"] = []
        result[paper_id]["qas"].append(deepcopy(question))
    return result


def evaluator(path):
    path = Path(path)
    if sha(path) != EVALUATOR_SHA:
        raise ValueError("unchanged original standalone publisher scorer required")
    spec = importlib.util.spec_from_file_location("original_qasper_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prediction(text):
    obj = json.loads(text)
    if (
        not isinstance(obj, dict)
        or set(obj) != {"predicted_answer", "predicted_evidence"}
        or not isinstance(obj["predicted_answer"], str)
        or not isinstance(obj["predicted_evidence"], list)
        or not all(isinstance(item, str) for item in obj["predicted_evidence"])
    ):
        raise ValueError("native answer/evidence strings required, no repair")
    return obj


def score(module, gold, predictions):
    pred = {
        row["question_id"]: {
            "answer": row["predicted_answer"],
            "evidence": row["predicted_evidence"],
        }
        for row in predictions
    }
    references = module.get_answers_and_evidence(gold, False)
    if set(pred) - set(references) or len(pred) != len(predictions):
        raise ValueError("duplicate or unknown native prediction identity")
    return module.evaluate(references, pred)


def post(url, payload, timeout):
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=tuple(FILES), default="dev")
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1")
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Socket read timeout, not a performance deadline or generation budget",
    )
    args = parser.parse_args()
    endpoint = validate_endpoint(args.endpoint)
    data, tokenizer_root, output = (
        args.data_root.resolve(),
        args.tokenizer.resolve(),
        args.output.resolve(),
    )
    if (
        args.timeout <= 0
        or output.exists()
        or any(
            output.is_relative_to(root) or root.is_relative_to(output)
            for root in (data, tokenizer_root, args.evaluator.resolve().parent)
        )
    ):
        raise ValueError("positive timeout and fresh separate output required")
    papers, rows = originals(data, args.split)
    selected = select(rows, start=args.start, count=args.count, full=args.full)
    module = evaluator(args.evaluator)
    gold = subset_gold(papers, selected)
    from transformers import AutoTokenizer

    identity = local_tokenizer_files(tokenizer_root)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, local_files_only=True, trust_remote_code=False
    )
    prepared = []
    for index, (paper_id, question_index, question) in selected:
        prompt = tokenizer.apply_chat_template(
            messages(papers[paper_id], question),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
        if tokens <= 0 or tokens + 8192 > 131072:
            raise ValueError(
                "complete original context exceeds deployment; never truncate/swap"
            )
        prepared.append((index, paper_id, question_index, question, prompt, tokens))
    output.mkdir(mode=0o700, parents=True)

    def save(name, value):
        path = output / name
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        path.chmod(0o600)

    receipt = {
        "source_id": "QASPER",
        "status": "PREPARED_NOT_EXECUTED",
        "split": args.split,
        "selected_original_positions": [i for i, _ in selected],
        "model": args.model,
        "script_sha256": sha(__file__),
        "original_split_sha256": FILES[args.split][2],
        "publisher_evaluator_sha256": EVALUATOR_SHA,
        "tokenizer_files": identity,
        "max_tokens": 8192,
        "context_capacity": 131072,
        "socket_timeout_seconds": args.timeout,
        "budget_boundary": "Explicit complete answer/evidence generative fallback budget, not author LED answer-only generation or training replication",
        "text_evidence_only": False,
        "original_reference_annotations_in_input": False,
        "input_truncated": False,
        "image_pixels_inferred": False,
        "model_calls": 0,
        "cases": [],
        "source_checkbox_eligible": False,
    }
    save("selected-gold.private.json", gold)
    save("receipt-private.json", receipt)
    save(
        "preflight-private.json",
        [
            {
                "original_index": i,
                "paper_id": p,
                "question_index": q,
                "question_id": row["question_id"],
                "input_tokens": n,
            }
            for i, p, q, row, _, n in prepared
        ],
    )
    if args.prepare_only:
        verify_tokenizer_files(tokenizer_root, identity)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "selected": len(selected),
                    "input_tokens": [item[-1] for item in prepared],
                    "model_calls": 0,
                }
            )
        )
        return
    predictions = []
    try:
        for index, paper_id, question_index, question, prompt, tokens in prepared:
            payload = {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": 8192,
                "temperature": 0,
                "seed": 20261026,
                "stream": False,
            }
            save(f"case-{index:05d}-request.private.json", payload)
            receipt["model_calls"] += 1
            receipt["status"] = "RUNNING"
            save("receipt-private.json", receipt)
            response = post(endpoint + "/completions", payload, args.timeout)
            save(f"case-{index:05d}-response.private.json", response)
            choices, usage = response.get("choices"), response.get("usage", {})
            if (
                not isinstance(choices, list)
                or len(choices) != 1
                or not isinstance(choices[0].get("text"), str)
                or choices[0].get("finish_reason") not in {"stop", "length"}
                or usage.get("prompt_tokens") != tokens
                or type(usage.get("completion_tokens")) is not int
                or not 0 <= usage["completion_tokens"] <= 8192
            ):
                raise ValueError("native terminal response integrity failed")
            case = {
                "original_index": index,
                "paper_id": paper_id,
                "question_index": question_index,
                "question_id": question["question_id"],
                "finish_reason": choices[0]["finish_reason"],
                "usage": usage,
            }
            try:
                native = prediction(choices[0]["text"])
                predictions.append({"question_id": question["question_id"], **native})
                case["prediction_format"] = "VALID_NATIVE_FIELDS"
            except ValueError as error:
                case.update(
                    prediction_format="INVALID_RETAINED_NO_RETRY",
                    error_type=type(error).__name__,
                )
            receipt["cases"].append(case)
            save("receipt-private.json", receipt)
        verify_tokenizer_files(tokenizer_root, identity)
        originals(data, args.split)
        evaluator(args.evaluator)
        receipt["scores"] = score(module, gold, predictions)
        receipt["invalid_predictions_retained"] = len(selected) - len(predictions)
        path = output / "native-predictions.private.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        path.chmod(0o600)
        receipt["status"] = (
            "ORIGINAL_QUESTIONS_NATIVE_GENERATION_AND_PUBLISHER_ANSWER_EVIDENCE_SCORING_COMPLETED"
        )
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
