"""Extend the frozen issue #3 workload to a fixed token-length band."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

FILLER = "\n【参考材料续】缓存系统记录请求到达、分配和完成时间；该段仅描述性能指标。"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.target_tokens < 1000:
        parser.error("target-tokens must be at least 1000")

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line]
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))

    def length(content: str) -> int:
        # A fixed estimate for chat-template overhead. The serving receipt
        # retains the exact prompt_tokens value for every request.
        return len(tokenizer.encode(content).ids) + 30

    repetitions: dict[str, int] = {}
    for row in rows:
        group = row["group_id"]
        if group in repetitions:
            continue
        before, after = row["messages"][0]["content"].rsplit("【本轮问题】", 1)
        lower, upper = 0, 1400
        while lower < upper:
            middle = (lower + upper) // 2
            candidate = before + FILLER * middle + "【本轮问题】" + after
            if length(candidate) < args.target_tokens:
                lower = middle + 1
            else:
                upper = middle
        repetitions[group] = lower

    for row in rows:
        before, after = row["messages"][0]["content"].rsplit("【本轮问题】", 1)
        content = (
            before + FILLER * repetitions[row["group_id"]] + "【本轮问题】" + after
        )
        row["messages"][0]["content"] = content
        row["prompt"] = content
        row["metadata"]["long_context_extension_repetitions"] = repetitions[
            row["group_id"]
        ]
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )


if __name__ == "__main__":
    main()
