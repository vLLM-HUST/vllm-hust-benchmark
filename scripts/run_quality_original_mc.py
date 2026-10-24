"""Three original NYU QuALITY questions, scored by fixed lm-eval SCROLLS MC methods.

Not the SCROLLS dataset, a publisher trained-baseline reproduction, chat generation
or a performance test. Raw requests/responses stay in the private output directory.
Requires lm-eval==0.4.11 and the existing local model's tokenizer, not CPU model weights.
"""

import argparse
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlparse

DATA_SHA = "99852d874994078e4b4112b71ceca4dd35aa3a24ff6d3a35c051be25295b4fef"  # pragma: allowlist secret - original public dev checksum
TASK_SHA = "f7bb4e60f3b2a3424a53653a25013b1c7f8ebb45f134edae99f6cac945d890ad"  # pragma: allowlist secret - pinned public harness source checksum
REVISION = "f84977c40dbfef70c9cab48037b7becfc8e45f73"  # pragma: allowlist secret - publisher source revision


def require(condition, message):
    if not condition:
        raise ValueError(message)


def endpoint(value):
    value = value.rstrip("/")
    parsed = urlparse(value)
    require(
        parsed.scheme == "http"
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.path == "/v1/completions"
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment,
        "only local HTTP completions endpoints without credentials permitted",
    )
    return value


def select_original_questions(parents, indices=None, full=False):
    require(
        not (full and indices is not None),
        "full and explicit indices cannot be combined",
    )
    flattened = [
        (index, pi, qi, parent, question)
        for index, (pi, qi, parent, question) in enumerate(
            (pi, qi, parent, question)
            for pi, parent in enumerate(parents)
            for qi, question in enumerate(parent["questions"])
        )
    ]
    if full:
        return flattened, len(flattened)
    if indices is not None:
        require(
            bool(indices) and len(set(indices)) == len(indices),
            "distinct original indices required",
        )
        require(
            all(0 <= index < len(flattened) for index in indices),
            "original question index out of range",
        )
        return [flattened[index] for index in indices], len(flattened)
    chosen, seen, flat_index = [], set(), 0
    for pi, parent in enumerate(parents):
        for qi, question in enumerate(parent["questions"]):
            if len(chosen) < 3 and parent["article_id"] not in seen:
                chosen.append((flat_index, pi, qi, parent, question))
                seen.add(parent["article_id"])
            flat_index += 1
    require(len(chosen) == 3, "three distinct original articles required")
    return chosen, flat_index


def original_document(parent, question, normalize):
    choices = [normalize(choice) for choice in question["options"]]
    require(
        len(choices) == len(set(choices)) == 4 and all(choices),
        "four distinct original answer options required",
    )
    gold = question["gold_label"]
    require(
        type(gold) is int and gold in {1, 2, 3, 4},
        "original 1-based gold label required",
    )
    return {
        "text": parent["article"],
        "question": question["question"],
        "choices": choices,
        "gold": gold - 1,
    }


def model_input_document(document):
    return {key: document[key] for key in ("text", "question", "choices")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1/completions")
    parser.add_argument("--max-length", type=int, default=131072)
    parser.add_argument(
        "--question-index",
        type=int,
        action="append",
        help="Repeat preselected original flat question indices; default is historical distinct-article three",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="All original dev questions with complete articles",
    )
    args = parser.parse_args()
    if args.full and args.question_index is not None:
        parser.error("--full cannot be combined with --question-index")
    base_url = endpoint(args.endpoint)
    require(args.max_length > 0, "positive declared model context budget required")
    require(not args.output.exists(), "private output directory must not already exist")
    require(args.tokenizer.is_dir(), "explicit existing local tokenizer required")
    data = args.data_root / "QuALITY.v1.0.1.htmlstripped.dev"
    raw = data.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == DATA_SHA,
        "original publisher dev identity mismatch",
    )
    chosen, questions = select_original_questions(
        [json.loads(line) for line in raw.decode().splitlines()],
        args.question_index,
        args.full,
    )
    require(
        questions == 2086
        and (
            args.full
            or args.question_index is not None
            or [x[0] for x in chosen] == [0, 6, 15]
        ),
        "fixed original dev selection changed",
    )
    require(
        metadata.version("lm-eval") == "0.4.11",
        "requires reviewed lm-eval==0.4.11 environment",
    )
    import lm_eval
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    task_path = Path(lm_eval.__file__).parent / "tasks/scrolls/task.py"
    require(
        hashlib.sha256(task_path.read_bytes()).hexdigest() == TASK_SHA,
        "fixed MC prompt/request/scorer identity mismatch",
    )
    spec = importlib.util.spec_from_file_location("quality_pinned_mc", task_path)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    # Pure existing prompt/request/metric methods; do not run SCROLLS downloading,
    # deduplication or pruning. Original NYU file identity is checked separately.
    task = upstream.QuALITY.__new__(upstream.QuALITY)
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    receipt = {
        "source_id": "QUALITY",
        "selected_flat_question_indices": [x[0] for x in chosen],
        "full_dev_requested": args.full,
        "source_revision": REVISION,
        "source_file_sha256": DATA_SHA,
        "harness_version": "0.4.11",
        "harness_task_sha256": TASK_SHA,
        "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": args.model,
        "started_at_unix": time.time(),
        "status": "STARTED",
        "samples": [],
        "completed_option_calls": 0,
        "max_context_tokens": args.max_length,
        "tokenizer_sha256": {
            name: hashlib.sha256((args.tokenizer / name).read_bytes()).hexdigest()
            for name in ("config.json", "tokenizer.json", "tokenizer_config.json")
        },
        "protocol": "original NYU questions; full article; unchanged lm-eval SCROLLS MC methods, not SCROLLS data or publisher trained-baseline reproduction",
        "choice_rendering": "upstream whitespace normalization only",
        "gold_excluded_from_request_document": True,
        "chat_template": False,
        "input_truncation": False,
        "full_suite_claim": False,
        "performance_claim": False,
        "api_one_token_tail": "harness likelihood protocol uses echo=True/logprobs=1/max_tokens=1; the tail is not a generated task answer",
    }

    def save():
        (args.output / "receipt.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        model = LocalCompletionsAPI(
            model=args.model,
            base_url=base_url,
            tokenizer=str(args.tokenizer),
            tokenizer_backend="huggingface",
            tokenized_requests=True,
            max_length=args.max_length,
            truncate=False,
            num_concurrent=1,
            batch_size=1,
            max_retries=0,
            timeout=300,
        )
        for index, pi, qi, parent, question in chosen:
            document = original_document(
                parent, question, upstream.QuALITY._normalize_answer
            )
            inputs = model_input_document(document)
            context = task.doc_to_text(inputs)
            requests = task.construct_requests(
                inputs, context, apply_chat_template=False
            )
            require(len(requests) == 4, "all four original options must be scored")
            scores, calls = [], []
            for choice_index, request in enumerate(requests):
                context_tokens, answer_tokens = model._encode_pair(*request.args)
                tokens = context_tokens + answer_tokens
                require(
                    len(tokens) + 1 <= args.max_length,
                    "complete original input exceeds budget; truncation forbidden",
                )
                response = model.model_call(messages=[tokens], generate=False)
                private = (
                    args.output / f"question-{index}-choice-{choice_index}.private.json"
                )
                private.write_text(
                    json.dumps({"request_token_ids": tokens, "response": response}),
                    encoding="utf-8",
                )
                parsed = model.parse_logprobs(
                    response, tokens=[tokens], ctxlens=[len(context_tokens)]
                )
                require(
                    len(parsed) == 1 and math.isfinite(parsed[0][0]),
                    "missing complete finite option likelihood",
                )
                require(
                    len(response["choices"]) == 1
                    and response["usage"]["prompt_tokens"] == len(tokens),
                    "full original token IDs not preserved by server",
                )
                scores.append((float(parsed[0][0]), bool(parsed[0][1])))
                calls.append(
                    {
                        "choice": choice_index,
                        "prompt_tokens": len(tokens),
                        "scored_option_tokens": len(answer_tokens),
                        "finish_reason": response["choices"][0]["finish_reason"],
                    }
                )
                receipt["completed_option_calls"] += 1
                save()
            receipt["samples"].append(
                {
                    "flat_question_index": index,
                    "parent_row_index": pi,
                    "question_index": qi,
                    "question_id": question["question_unique_id"],
                    "article_id": parent["article_id"],
                    "metrics": task.process_results(document, scores),
                    "requests": calls,
                }
            )
            save()
        require(
            hashlib.sha256(data.read_bytes()).hexdigest() == DATA_SHA,
            "original file changed during execution",
        )
        receipt["status"] = "ORIGINAL_QUESTIONS_MC_EXECUTED_HARNESS_SCORED"
    except BaseException as error:
        receipt.update(status="FAILED", error_type=type(error).__name__)
        raise
    finally:
        receipt["finished_at_unix"] = time.time()
        save()
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
