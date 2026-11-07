"""Recheck original DRAGON saved inputs, native embeddings and author scores.

No model loading, NPU initialization, retrieval execution, API call or task repair.
Original text/IDs and the full prompts/results remain external and private.
"""

import argparse
import ast
from collections import defaultdict
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys

from prepare_jsonschemabench_author_source import verify_source
from run_dragon_original import (
    CONFIG,
    generation_parameters,
    original_system_prompt,
    reader_generation_config,
    reference_subset,
    sha,
)


def original_template(author, tokenizer):
    tree = ast.parse((author / "lib/src/rag_bench/baseline.py").read_text("utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "init_generation"
    )
    assignments = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "messages"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("one unchanged original messages template required")

    class SystemConstant(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id != "llm_prompt":
                raise ValueError(
                    "unexpected executable expression in original messages"
                )
            return ast.Constant(original_system_prompt(author))

    messages = ast.literal_eval(SystemConstant().visit(assignments[0].value))
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def bind_context(context, ids, choices):
    """Find exact ordered original chunks; no substring rewrite or deduplication."""

    @lru_cache(maxsize=None)
    def match(index, offset):
        if index == len(ids):
            return () if offset == len(context) else None
        for chunk in choices.get(ids[index], []):
            text = ("\n\n" if index else "") + chunk
            if context.startswith(text, offset):
                rest = match(index + 1, offset + len(text))
                if rest is not None:
                    return (chunk,) + rest
        return None

    result = match(0, 0)
    if result is None:
        raise ValueError(
            "reader context is not exact original chunks in returned ID order"
        )
    return result


def verify_vectors(record, expected_texts, dimension):
    if record["texts"] != expected_texts or record["encode_kwargs"] != {}:
        raise ValueError(
            "original standalone embedding input order/instruction differs"
        )
    vectors = record["embeddings"]
    if len(vectors) != len(expected_texts) or any(
        len(row) != dimension for row in vectors
    ):
        raise ValueError("original encoder dimensions or batch rows differ")
    if any(
        type(value) not in (int, float) or not math.isfinite(value)
        for row in vectors
        for value in row
    ):
        raise ValueError("finite original native vectors required")


def verify(args):
    native = args.native_root.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(native) or not output.parent.is_dir():
        raise ValueError("new external audit receipt with existing parent required")
    receipt = json.loads((native / "native-public.json").read_bytes())
    if (
        receipt["status"]
        != "FIXED_ORIGINAL_RETRIEVAL_READER_AND_OFFICIAL_SCORING_COMPLETE_AUDIT_PENDING"
    ):
        raise ValueError(
            "terminal original native run required; preserve partial failures"
        )
    author = verify_source(args.author_root, CONFIG / "dragon-author-source.json")
    sys.path.insert(0, str(author / "lib/src"))
    from datasets import Dataset, DatasetDict
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rag_bench import evaluator
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from vllm_hust_workload_data.dragon import original_rows

    data = args.data_root.resolve(strict=True)
    source_rows = list(
        original_rows(data / "public_questions/data/train-00000-of-00001.parquet")
    )
    ids = receipt["original_selected_question_ids"]
    if ids != [
        row["original_question"]["id"] for row in source_rows[: len(ids)]
    ] or len(ids) not in (3, len(source_rows)):
        raise ValueError("fixed first3 or full original source required")
    if any(
        receipt[key] != value
        for key, value in {
            "top_k": 5,
            "chunk_size_characters": 1000,
            "chunk_overlap_characters": 100,
            "search_type": "mmr",
        }.items()
    ):
        raise ValueError("original standalone retrieval profile differs")
    texts = Dataset.from_parquet(
        str(data / "public_texts/data/train-00000-of-00001.parquet")
    )
    references = Dataset.from_parquet(
        str(data / "reference_qa/data/train-00000-of-00001.parquet")
    )
    mapping_rows = Dataset.from_parquet(
        str(data / "reference_texts/data/train-00000-of-00001.parquet")
    )
    split = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    ).split_documents(
        [
            Document(id=index, page_content=row["text"], metadata={"id": row["id"]})
            for index, row in enumerate(texts)
        ]
    )
    if (
        len(texts) != receipt["original_corpus_documents"]
        or len(split) != receipt["indexed_chunks"]
    ):
        raise ValueError("complete original corpus/index required")
    chunks = defaultdict(list)
    for item in split:
        chunks[item.metadata["id"]].append(item.page_content)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.reader_root), local_files_only=True, trust_remote_code=False
    )
    template = original_template(author, tokenizer)
    config, config_source = reader_generation_config(
        args.reader_root, GenerationConfig, AutoConfig
    )
    parameters = generation_parameters(config, receipt["reader_model"])
    if (
        parameters != receipt["reader_generation_parameters"]
        or config_source != receipt["reader_generation_config_source"]
    ):
        raise ValueError("unchanged original reader generation configuration required")
    if sha(args.reader_root / "config.json") != receipt["reader_model_config_sha256"]:
        raise ValueError("original reader model configuration changed")
    answers = json.loads((native / "answers-private.json").read_bytes())
    if set(answers) != set(map(str, ids)) or receipt["reader_calls"] != len(ids):
        raise ValueError("all and only selected original reader results required")
    finishes = []
    for index, row in enumerate(source_rows[: len(ids)]):
        question = row["original_question"]
        answer = answers[str(question["id"])]
        request = json.loads(
            (native / f"reader-{index:04d}-request-private.json").read_bytes()
        )
        response = json.loads(
            (native / f"reader-{index:04d}-response-private.json").read_bytes()
        )
        expected = dict(parameters, prompt=request["prompt"])
        if request != expected or len(response["choices"]) != 1:
            raise ValueError("saved complete original request or response differs")
        choice = response["choices"][0]
        if choice["text"] != answer["model_answer"] or choice["finish_reason"] not in {
            "stop",
            "length",
        }:
            raise ValueError("saved native response and author answer differ")
        marker = "__DRAGON_ORIGINAL_CONTEXT_BINDING__"
        if marker in template or marker in question["question"]:
            raise ValueError("context marker collision")
        formatted = template.format(context=marker, input=question["question"])
        prefix, suffix = formatted.split(marker)
        if not request["prompt"].startswith(prefix) or not request["prompt"].endswith(
            suffix
        ):
            raise ValueError("original system/question/template changed")
        context = request["prompt"][len(prefix) : len(request["prompt"]) - len(suffix)]
        if len(answer["found_ids"]) != 5:
            raise ValueError("complete original top5 returned documents required")
        bind_context(context, tuple(answer["found_ids"]), chunks)
        finishes.append(choice["finish_reason"])
    cache_files = sorted(native.glob("embedding-batch-*-private.json"))
    batches = [
        [item.page_content for item in split[index : index + 1000]]
        for index in range(0, len(split), 1000)
    ]
    batches.extend(
        [[row["original_question"]["question"]] for row in source_rows[: len(ids)]]
    )
    if len(cache_files) != len(batches) or receipt["npu_embedding_encode_calls"] != len(
        batches
    ):
        raise ValueError("complete original native embedding batch evidence required")
    for file, batch in zip(cache_files, batches, strict=True):
        verify_vectors(json.loads(file.read_bytes()), batch, 1024)
    selected = Dataset.from_list(reference_subset(list(references), ids))
    mapping = {row["public_id"]: row["id"] for row in mapping_rows}
    scores = evaluator.evaluate_rag_results(
        answers, DatasetDict(train=selected), mapping
    ).to_dict()
    if scores != json.loads((native / "official-scores-private.json").read_bytes()):
        raise ValueError("unchanged author scoring differs from saved results")
    for key, name in (
        ("answers_sha256", "answers-private.json"),
        ("scores_sha256", "official-scores-private.json"),
    ):
        if receipt[key] != sha(native / name):
            raise ValueError("saved original answer/score identity differs")
    result = {
        "source_id": "DRAGON_PUBLIC",
        "status": "ORIGINAL_SAVED_PROTOCOL_NATIVE_EMBEDDINGS_AND_AUTHOR_SCORES_VERIFIED_NO_MODEL_CALLS",
        "original_question_ids": ids,
        "original_corpus_documents": len(texts),
        "original_chunks": len(split),
        "original_requests_responses_and_complete_prompt_contexts_verified": True,
        "reference_answers_not_added_to_original_inputs": True,
        "embedding_batches_verified": len(batches),
        "reader_finish_reasons": finishes,
        "official_scores_equal_saved": True,
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "boundary": "Saved actual native outputs rechecked; not NPU re-execution, publisher model-score reproduction, performance evidence or whole-source delivery closure",
    }
    os.umask(0o077)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native-root", "author-root", "data-root", "reader-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(verify(parser.parse_args())))


if __name__ == "__main__":
    main()
