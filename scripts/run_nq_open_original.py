"""Original NQ-open through pinned lm-eval generation/filter/Exact Match.

Publisher question/answer files stay unchanged; local JSON paths replace only
the harness data acquisition. This zero-shot closed-book profile is not DPR,
REALM, full Natural Questions document QA, or a paper-model replication.
"""

import argparse
import hashlib
from functools import partial
from importlib import metadata
import json
from pathlib import Path
from urllib.parse import urlparse

from run_toolace_original_continuation import (
    local_tokenizer_files,
    verify_tokenizer_files,
)

REVISION = "fb26a3073b1fe636c97302890a27b491d6530130"  # pragma: allowlist secret - public original dataset revision
TASK_SHA = "c1c66deceebfc4bbd70bd9a78d9710795e95a62380207974b44855094a936c91"  # pragma: allowlist secret - upstream lm-eval v0.4.11 nq_open.yaml
FILES = {
    "train": (
        "nq_open/NQ-open.train.jsonl",
        8522298,
        87925,
        "0c93aaf13b12f855628489665e4ef85ffb8573ad444646f11f4fabdc57a0aabf",  # pragma: allowlist secret - original publisher file
    ),
    "validation": (
        "nq_open/NQ-open.dev.jsonl",
        391316,
        3610,
        "f15567f38099f3615f5b8a685c0aef449c11ad90d3da3735e8d1b98115b40616",  # pragma: allowlist secret - original publisher file
    ),
}
MAX_GEN_TOKS = 256  # Pinned lm-eval0.4.11 local API default, not a reduced task budget.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(*, start=0, count=3, full=False):
    if full:
        if start != 0 or count != 3:
            raise ValueError("full cannot mix with explicit range")
        return list(range(3610))
    if (
        type(start) is not int
        or type(count) is not int
        or start < 0
        or count < 3
        or start + count > 3610
    ):
        raise ValueError("preselect at least three original dev positions")
    return list(range(start, start + count))


def originals(root):
    root = Path(root).resolve()
    paths, data = {}, {}
    for split, (name, size, count, expected_sha) in FILES.items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root)
            or path.stat().st_size != size
            or sha(path) != expected_sha
        ):
            raise ValueError("unchanged pinned original NQ-open file required")
        with path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != count or any(
            not isinstance(row, dict)
            or set(row) != {"question", "answer"}
            or not isinstance(row["question"], str)
            or not isinstance(row["answer"], list)
            or not row["answer"]
            or not all(isinstance(answer, str) for answer in row["answer"])
            for row in rows
        ):
            raise ValueError("complete original question/answer aliases required")
        paths[split], data[split] = str(path), rows
    return paths, data


def local_config(original, paths):
    # Keep task name, prompts, stop strings, filters, references and metric intact.
    result = dict(original)
    result.update(
        dataset_path="json", dataset_name=None, dataset_kwargs={"data_files": paths}
    )
    return result


def verify_native_samples(samples, selected, original_rows, contexts):
    if len(samples) != len(selected) or {s["doc_id"] for s in samples} != set(selected):
        raise ValueError("native samples differ from fixed original selection")
    for sample in samples:
        i = sample["doc_id"]
        if (
            sample["doc"] != original_rows[i]
            or sample["arguments"][0][0] != contexts[i]
        ):
            raise ValueError("native request/document changed original context")


def bind_prompt_options(model, *, apply_chat_template, disable_thinking):
    """Model formatting only; never change the task's text/stop/filter/metric."""
    if disable_thinking and not apply_chat_template:
        raise ValueError("disable-thinking requires the explicit native chat mode")
    if disable_thinking:
        if "enable_thinking" not in str(model.tokenizer.chat_template):
            raise ValueError("model template has no reviewed enable_thinking option")
        model.tokenizer.apply_chat_template = partial(
            model.tokenizer.apply_chat_template, enable_thinking=False
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:19466/v1/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--concurrency", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Official harness native-model chat formatting; distinct from the default plain-text profile",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Bind the model's native enable_thinking=False option; requires chat mode and a supporting template",
    )
    args = parser.parse_args()
    if args.disable_thinking and not args.apply_chat_template:
        parser.error("--disable-thinking requires --apply-chat-template")
    selected = select(start=args.start, count=args.count, full=args.full)
    endpoint = urlparse(args.endpoint)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "localhost"}
        or endpoint.path != "/v1/completions"
        or endpoint.query
        or endpoint.fragment
        or endpoint.username
        or endpoint.password
    ):
        raise ValueError("explicit local original completions endpoint required")
    if metadata.version("lm-eval") != "0.4.11" or args.max_length <= MAX_GEN_TOKS:
        raise ValueError(
            "reviewed lm-eval0.4.11 and sufficient original context required"
        )
    output = args.output.resolve()
    if output.exists() or any(
        output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output)
        for p in (args.data_root, args.tokenizer)
    ):
        raise ValueError("fresh private output separate from assets/model required")
    paths, data = originals(args.data_root)
    tokenizer_files = local_tokenizer_files(args.tokenizer)
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.api.registry import get_model
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.utils import load_yaml_config
    import lm_eval.models  # noqa: F401 - registers the official model backends

    original_yaml = Path(lm_eval.__file__).parent / "tasks/nq_open/nq_open.yaml"
    if sha(original_yaml) != TASK_SHA:
        raise ValueError("original nq_open generation/filter/metric config changed")
    config = local_config(load_yaml_config(str(original_yaml)), paths)
    task = ConfigurableTask(config=config)
    if (
        list(task.training_docs()) != data["train"]
        or list(task.validation_docs()) != data["validation"]
    ):
        raise ValueError("harness JSON loading altered original rows or order")
    output.mkdir(mode=0o700, parents=True)
    receipt = {
        "source_id": "NATURALQUESTIONS",
        "profile": "nq_open_original_dev_closed_book_zero_shot_native_chat"
        if args.apply_chat_template
        else "nq_open_original_dev_closed_book_zero_shot_plain",
        "source_revision": REVISION,
        "status": "PREPARING_NATIVE_NOT_COMPLETE",
        "selected_original_dev_positions": selected,
        "full_dev_selected": args.full,
        "source_checkbox_eligible": False,
        "native_execution_complete": False,
        "model": args.model,
        "context_capacity": args.max_length,
        "max_gen_toks": MAX_GEN_TOKS,
        "concurrency": args.concurrency,
        "harness_version": "0.4.11",
        "task_config_sha256": TASK_SHA,
        "tokenizer_files": tokenizer_files,
        "input_truncated": False,
        "chat_template_applied": args.apply_chat_template,
        "chat_template_parameters": {"enable_thinking": False}
        if args.disable_thinking
        else {},
        "num_fewshot": 0,
        "scoring": "Original lm-eval nq_open config Exact Match over complete original answer aliases; not a reproduction of the publisher TensorFlow evaluation CLI",
        "model_or_retrieval_paper_replication": False,
        "performance_claim": False,
    }

    def save():
        (output / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )

    save()
    try:
        model = get_model("local-completions").create_from_arg_obj(
            {
                "model": args.model,
                "base_url": args.endpoint,
                "tokenizer": str(args.tokenizer.resolve()),
                "tokenizer_backend": "huggingface",
                "tokenized_requests": True,
                "num_concurrent": args.concurrency,
                "max_retries": 0,
                "max_length": args.max_length,
                "max_gen_toks": MAX_GEN_TOKS,
                "truncate": False,
                "timeout": 300,
            }
        )
        bind_prompt_options(
            model,
            apply_chat_template=args.apply_chat_template,
            disable_thinking=args.disable_thinking,
        )
        contexts, token_lengths = {}, []
        for i in selected:
            row = data["validation"][i]
            if task.doc_to_target(row) != row["answer"]:
                raise ValueError("harness lost original answer aliases")
            context = task.fewshot_context(
                row,
                num_fewshot=0,
                apply_chat_template=args.apply_chat_template,
                fewshot_as_multiturn=False,
                chat_template=model.apply_chat_template,
            )
            encoded = model.tok_encode(context, add_special_tokens=False)
            if not encoded or len(encoded) + MAX_GEN_TOKS > args.max_length:
                raise ValueError(
                    "original complete context exceeds capacity; no truncation"
                )
            contexts[i] = context
            token_lengths.append(len(encoded))
        receipt.update(
            status="RUNNING_ORIGINAL_NATIVE_GENERATION",
            preflight_prompt_token_min=min(token_lengths),
            preflight_prompt_token_max=max(token_lengths),
        )
        save()
        result = evaluator.simple_evaluate(
            model=model,
            tasks=[task],
            samples={"nq_open": selected},
            num_fewshot=0,
            batch_size=1,
            bootstrap_iters=0,
            log_samples=True,
            apply_chat_template=args.apply_chat_template,
            fewshot_as_multiturn=False,
            use_cache=str(output / "native-response-cache"),
        )
        if result is None:
            raise ValueError("native harness returned no result")
        samples = result["samples"]["nq_open"]
        verify_native_samples(samples, selected, data["validation"], contexts)
        originals(args.data_root)
        verify_tokenizer_files(args.tokenizer, tokenizer_files)
        private = output / "harness-results-private.json"
        private.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        receipt.update(
            status="ORIGINAL_DEV_NATIVE_GENERATION_AND_HARNESS_SCORING_COMPLETE",
            native_execution_complete=True,
            actual_samples=len(samples),
            metrics=result["results"],
            private_result_sha256=sha(private),
        )
    except Exception as error:
        receipt.update(status="FAILED_NOT_COMPLETE", error_type=type(error).__name__)
        raise
    finally:
        save()
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in ("source_id", "status", "actual_samples", "metrics")
            }
        )
    )


if __name__ == "__main__":
    main()
