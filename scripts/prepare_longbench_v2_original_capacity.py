"""Prepare explicit original LongBench-v2 capacity bindings without model requests.

Use the native runner's isolated environment and local, publisher-verified model
tokenizer. The original get_pred constructs each prompt, then a BaseException
sentinel stops execution before query_llm, truncation, generation or scoring.
This is preparation, not native benchmark execution.
"""

import argparse
import copy
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile


DATA_SHA = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"  # pragma: allowlist secret - original data checksum
ARCHIVE_SHA = "2f92f9953e3dd2fd3915a132a0a4cb53acb879af6ef1e98962129a6f3fe262bb"  # pragma: allowlist secret - original source archive checksum


class PromptCaptured(BaseException):
    """Exit before any original inference or official output is produced."""


def original_indices(indices=None, *, full=False):
    if full:
        if indices is not None:
            raise ValueError("full cannot be combined with explicit row indices")
        return list(range(503))
    if not isinstance(indices, list) or not 1 <= len(indices) <= 503:
        raise ValueError("explicit original row indices required")
    if any(type(index) is not int or not 0 <= index < 503 for index in indices):
        raise ValueError("invalid original row index")
    if len(set(indices)) != len(indices) or indices != sorted(indices):
        raise ValueError("indices must be unique and in original order")
    return indices


def capture_original_prompt(module, item, model, model_path, tokenizer):
    """Run unchanged upstream prompt construction, never fabricate an answer."""
    original_query = module.query_llm
    original_client = module.OpenAI
    original_loader = module.AutoTokenizer
    captured = []
    client_sentinel = object()

    def loader(path, *, trust_remote_code):
        if path != model_path or trust_remote_code is not True:
            raise ValueError("unexpected original tokenizer loading contract")
        return tokenizer

    def client(*, base_url, api_key):
        if base_url != module.URL:
            raise ValueError("unexpected original client configuration")
        return client_sentinel

    def query(prompt, alias, actual_tokenizer, client, *, temperature, max_new_tokens):
        if (
            alias != model
            or actual_tokenizer is not tokenizer
            or client is not client_sentinel
            or temperature != 0.1
            or max_new_tokens != 128
            or not isinstance(prompt, str)
        ):
            raise ValueError("original non-CoT prompt/query contract changed")
        captured.append(prompt)
        raise PromptCaptured()

    module.AutoTokenizer = SimpleNamespace(from_pretrained=loader)
    module.OpenAI = client
    module.query_llm = query
    try:
        try:
            module.get_pred(
                [copy.deepcopy(item)],
                SimpleNamespace(model=model, rag=0, no_context=False, cot=False),
                None,
            )
        except PromptCaptured:
            pass
        if len(captured) != 1:
            raise ValueError("original prompt was not captured exactly once")
        return captured[0]
    finally:
        module.query_llm = original_query
        module.OpenAI = original_client
        module.AutoTokenizer = original_loader


def capacity_row(item, index, prompt, tokenizer, *, model_max_length=131072):
    if type(model_max_length) is not int or not 128 < model_max_length <= 2097152:
        raise ValueError("explicit deployment window must be 129..2097152 tokens")
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int for token in tokens)
    ):
        raise ValueError("expected complete tokenized single-message chat input")
    return {
        "row_index": index,
        "id": item["_id"],
        "length_label": item["length"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "full_prompt_tokens": len(tokens),
        "status": (
            "FULL_INPUT_FITS"
            if len(tokens) + 128 <= model_max_length
            else "RESOURCE_LIMIT_FULL_CONTEXT"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="workload-qwen35-27b-128k-nonthinking")
    parser.add_argument("--model-max-length", type=int, default=131072)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--row-index", action="append", type=int)
    selection.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if not 128 < args.model_max_length <= 2097152:
        parser.error("deployment window must be 129..2097152 tokens")
    args.output = args.output.resolve()
    indices = original_indices(args.row_index, full=args.full)
    if "gpt" in args.model or "o1" in args.model:
        parser.error(
            "this local-tokenizer profile cannot use upstream tiktoken aliases"
        )
    if hashlib.sha256(args.data_file.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError("original LongBench-v2 data identity mismatch")
    if hashlib.sha256(args.official_archive.read_bytes()).hexdigest() != ARCHIVE_SHA:
        raise ValueError("original LongBench source archive identity mismatch")
    rows = json.loads(args.data_file.read_text("utf-8"))
    if not isinstance(rows, list) or len(rows) != 503:
        raise ValueError("expected the complete original503-row file")
    model_path = str(args.tokenizer.resolve())
    tokenizer_file = args.tokenizer / "tokenizer.json"
    tokenizer_sha = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    os.umask(0o077)
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    with zipfile.ZipFile(args.official_archive) as archive:
        archive.extractall(args.output / "official-source")
    source = args.output / "official-source" / "THUDM-LongBench-2e00731"
    previous = Path.cwd()
    os.chdir(source)
    try:
        spec = importlib.util.spec_from_file_location(
            "official_capacity_pred", source.resolve() / "pred.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(previous)
    module.model_map = {args.model: model_path}
    selected = []
    for index in indices:
        prompt = capture_original_prompt(
            module, rows[index], args.model, model_path, tokenizer
        )
        selected.append(
            capacity_row(
                rows[index],
                index,
                prompt,
                tokenizer,
                model_max_length=args.model_max_length,
            )
        )
    if hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != tokenizer_sha:
        raise ValueError("local tokenizer changed during preparation")
    receipt = {
        "source_id": "LONGBENCH_V2",
        "model": args.model,
        "model_max_length": args.model_max_length,
        "rule": "explicit original indices frozen before capacity checks; no filtering, replacement, truncation or model outcome selection",
        "selected": selected,
        "ready_for_execution": all(
            row["status"] == "FULL_INPUT_FITS" for row in selected
        ),
        "original_data_sha256": DATA_SHA,
        "official_archive_sha256": ARCHIVE_SHA,
        "tokenizer_json_sha256": tokenizer_sha,
        "packages": {
            name: metadata.version(name) for name in ("transformers", "tokenizers")
        },
        "prompt_construction": "unchanged official get_pred; stopped before query_llm",
        "chat_template_kwargs": {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "return_dict": False,
        },
        "original_output_budget": 128,
        "model_generation_calls": 0,
        "official_scoring_executed": False,
        "native_benchmark_executed": False,
        "source_completion_checked": False,
    }
    (args.output / "capacity-selection.private.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "prepared_original_tasks": len(selected),
                "ready_for_execution": receipt["ready_for_execution"],
                "model_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
