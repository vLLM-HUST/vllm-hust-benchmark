#!/usr/bin/env python3
"""Convert an EvoScientist workload trace into vllm-hust benchmark dataset formats.

Reads the JSONL trace produced by run_evoscientist_trace.py and outputs:
1. Custom JSONL dataset (for `vllm bench serve --dataset-name custom`)
2. ShareGPT-format JSON (for `vllm bench serve --dataset-name sharegpt`)

Usage:
    python scripts/convert_trace_to_workload.py [--input traces/in.jsonl] [--output-dir traces/]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).parent / "traces" / "evoscientist-research-trace.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "traces"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert EvoScientist trace to benchmark workload formats"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input JSONL trace file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for converted workload files",
    )
    parser.add_argument(
        "--min-input-tokens",
        type=int,
        default=10,
        help="Minimum input tokens to include an entry (filter noise)",
    )
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=5,
        help="Minimum output tokens to include an entry (filter noise)",
    )
    return parser.parse_args()


def load_trace(path: Path) -> list[dict[str, Any]]:
    """Load trace entries from JSONL file."""
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {line_num}: {e}")
    return entries


def messages_to_prompt_text(messages: Any) -> str:
    """Convert message list(s) to a single prompt string for custom format.

    Extracts the last user message as the prompt, or concatenates all messages
    with role prefixes for context.
    """
    if not messages:
        return ""

    # Handle nested list (batch of message lists)
    if isinstance(messages, list) and messages and isinstance(messages[0], list):
        messages = messages[0]

    # If it's a list of strings (old-style prompts), join them
    if isinstance(messages, list) and messages and isinstance(messages[0], str):
        return "\n".join(messages)

    # List of message dicts
    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        # Build the full conversation as prompt context
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]: {content}")
            elif role == "human":
                parts.append(f"[User]: {content}")
            elif role == "ai":
                parts.append(f"[Assistant]: {content}")
            elif role == "tool":
                # Include tool results as context
                name = msg.get("name", "tool")
                parts.append(f"[Tool:{name}]: {content[:500]}")
            else:
                parts.append(f"[{role}]: {content}")
        return "\n".join(parts)

    return str(messages)


def extract_last_user_message(messages: Any) -> str:
    """Extract the last user/human message from a message list."""
    if not messages:
        return ""

    # Flatten nested batch
    if isinstance(messages, list) and messages and isinstance(messages[0], list):
        messages = messages[0]

    if isinstance(messages, list) and messages and isinstance(messages[0], dict):
        # Find last human/user message (LangChain uses "human", OpenAI uses "user")
        for msg in reversed(messages):
            if msg.get("role") in ("human", "user"):
                return msg.get("content", "")
        # Fallback to last message
        return messages[-1].get("content", "") if messages else ""

    if isinstance(messages, list) and messages and isinstance(messages[0], str):
        return messages[-1]

    return str(messages)


def convert_to_custom_jsonl(
    entries: list[dict[str, Any]],
    min_input_tokens: int,
    min_output_tokens: int,
) -> list[dict[str, Any]]:
    """Convert trace entries to custom JSONL format.

    Format: {"prompt": "<text>", "output_tokens": N}
    """
    results = []
    for entry in entries:
        input_tokens = entry.get("input_tokens", 0)
        output_tokens = entry.get("output_tokens", 0)

        # Filter out tiny entries (likely tool calls or metadata)
        if input_tokens < min_input_tokens or output_tokens < min_output_tokens:
            continue

        prompt = messages_to_prompt_text(entry.get("prompt_messages"))
        if not prompt:
            continue

        results.append({
            "prompt": prompt,
            "output_tokens": output_tokens,
        })

    return results


def convert_to_sharegpt(
    entries: list[dict[str, Any]],
    min_input_tokens: int,
    min_output_tokens: int,
) -> list[dict[str, Any]]:
    """Convert trace entries to ShareGPT conversation format.

    Groups sequential entries into multi-turn conversations based on
    parent_run_id relationships. Each conversation has alternating
    human/gpt turns.

    Format: [{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}]
    """
    conversations = []

    for entry in entries:
        input_tokens = entry.get("input_tokens", 0)
        output_tokens = entry.get("output_tokens", 0)

        if input_tokens < min_input_tokens or output_tokens < min_output_tokens:
            continue

        user_msg = extract_last_user_message(entry.get("prompt_messages"))
        completion = entry.get("completion", "")

        if not user_msg or not completion:
            continue

        conversations.append({
            "conversations": [
                {"from": "human", "value": user_msg},
                {"from": "gpt", "value": completion},
            ]
        })

    return conversations


def print_stats(entries: list[dict], custom: list[dict], sharegpt: list[dict]) -> None:
    """Print summary statistics about the converted workload."""
    print("\n" + "=" * 60)
    print("WORKLOAD TRACE CONVERSION SUMMARY")
    print("=" * 60)

    print(f"\nRaw trace entries:     {len(entries)}")
    print(f"Custom JSONL entries:  {len(custom)}")
    print(f"ShareGPT entries:      {len(sharegpt)}")

    if custom:
        output_tokens = [e["output_tokens"] for e in custom]
        prompt_lens = [len(e["prompt"]) for e in custom]
        print(f"\nOutput tokens:")
        print(f"  min:    {min(output_tokens)}")
        print(f"  max:    {max(output_tokens)}")
        print(f"  mean:   {statistics.mean(output_tokens):.0f}")
        print(f"  median: {statistics.median(output_tokens):.0f}")
        print(f"\nPrompt length (chars):")
        print(f"  min:    {min(prompt_lens)}")
        print(f"  max:    {max(prompt_lens)}")
        print(f"  mean:   {statistics.mean(prompt_lens):.0f}")
        print(f"  median: {statistics.median(prompt_lens):.0f}")

    if entries:
        latencies = [e.get("latency_ms", 0) for e in entries if e.get("latency_ms", 0) > 0]
        if latencies:
            print(f"\nLatency (ms):")
            print(f"  min:    {min(latencies):.0f}")
            print(f"  max:    {max(latencies):.0f}")
            print(f"  mean:   {statistics.mean(latencies):.0f}")
            print(f"  median: {statistics.median(latencies):.0f}")
            print(f"  total:  {sum(latencies)/1000:.1f}s")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input trace file not found: {args.input}")
        print("Run 'python scripts/run_evoscientist_trace.py' first to generate a trace.")
        sys.exit(1)

    # Load trace
    print(f"[INFO] Loading trace from: {args.input}")
    entries = load_trace(args.input)
    if not entries:
        print("[ERROR] No entries found in trace file.")
        sys.exit(1)
    print(f"[INFO] Loaded {len(entries)} trace entries")

    # Convert to custom JSONL
    custom_entries = convert_to_custom_jsonl(
        entries, args.min_input_tokens, args.min_output_tokens
    )

    # Convert to ShareGPT
    sharegpt_entries = convert_to_sharegpt(
        entries, args.min_input_tokens, args.min_output_tokens
    )

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    custom_path = args.output_dir / "evoscientist-workload-custom.jsonl"
    with open(custom_path, "w", encoding="utf-8") as f:
        for entry in custom_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[OK] Custom JSONL written: {custom_path} ({len(custom_entries)} entries)")

    sharegpt_path = args.output_dir / "evoscientist-workload-sharegpt.json"
    with open(sharegpt_path, "w", encoding="utf-8") as f:
        json.dump(sharegpt_entries, f, ensure_ascii=False, indent=2)
    print(f"[OK] ShareGPT JSON written: {sharegpt_path} ({len(sharegpt_entries)} entries)")

    # Print stats
    print_stats(entries, custom_entries, sharegpt_entries)

    print(f"\n[SUCCESS] Conversion complete.")
    print(f"  Custom:   {custom_path}")
    print(f"  ShareGPT: {sharegpt_path}")
    print(f"\nTo benchmark with vllm-hust:")
    print(f"  vllm bench serve --dataset-name custom --dataset-path {custom_path} \\")
    print(f"    --model <model> --num-prompts {len(custom_entries)}")


if __name__ == "__main__":
    main()
