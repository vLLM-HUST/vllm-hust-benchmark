#!/usr/bin/env python3
"""Audit whether a JSONL prompt set can reuse complete prefix-cache pages."""

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--page-tokens", type=int, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    texts = [json.loads(line)["prompt"] for line in args.dataset.open()]
    prompts = []
    for text in texts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(tokenizer.encode(rendered, add_special_tokens=False))

    lengths = [len(prompt) for prompt in prompts]
    adjacent = [
        common_prefix_length(prompts[index - 1], prompts[index])
        for index in range(1, len(prompts))
    ]
    seen_pages: set[tuple[int, ...]] = set()
    hit_tokens = []
    for prompt in prompts:
        pages = [
            tuple(prompt[:end])
            for end in range(args.page_tokens, len(prompt) + 1, args.page_tokens)
        ]
        hit_tokens.append(
            sum(args.page_tokens for page in pages if page in seen_pages)
        )
        seen_pages.update(pages)

    print(
        json.dumps(
            {
                "count": len(prompts),
                "page_tokens": args.page_tokens,
                "token_lengths": {
                    "min": min(lengths),
                    "median": statistics.median(lengths),
                    "max": max(lengths),
                    "mean": statistics.mean(lengths),
                    "at_least_one_page": sum(
                        length >= args.page_tokens for length in lengths
                    ),
                },
                "adjacent_common_prefix_tokens": {
                    "min": min(adjacent),
                    "median": statistics.median(adjacent),
                    "max": max(adjacent),
                    "mean": statistics.mean(adjacent),
                    "at_least_one_page": sum(
                        length >= args.page_tokens for length in adjacent
                    ),
                },
                "optimistic_sequential_cache": {
                    "requests_with_hits": sum(hit > 0 for hit in hit_tokens),
                    "hit_tokens": sum(hit_tokens),
                    "query_tokens": sum(lengths),
                    "hit_ratio": sum(hit_tokens) / sum(lengths),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
