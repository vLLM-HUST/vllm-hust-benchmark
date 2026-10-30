"""Replay a JSONL similarity workload against an OpenAI completions server.

The same ordered input must be replayed separately against a baseline server
and a SimLLM server. The first item in each group warms the task cache.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import statistics
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen


def complete(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    capture_text: bool = False,
    messages: list[dict] | None = None,
) -> dict:
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if messages is None:
        payload["prompt"] = prompt
        endpoint = "completions"
    else:
        payload["messages"] = messages
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        endpoint = "chat/completions"
    body = json.dumps(payload).encode()
    request = Request(
        f"{url.rstrip('/')}/v1/{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first_token = None
    usage = {}
    request_id = None
    output_parts: list[str] = []
    with urlopen(request, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            request_id = request_id or event.get("id")
            if event.get("usage"):
                usage = event["usage"]
            parts = [
                (
                    choice.get("delta", {}).get("content")
                    if messages is not None
                    else choice.get("text")
                )
                or ""
                for choice in event.get("choices", [])
            ]
            if any(parts):
                first_token = first_token or time.monotonic()
                if capture_text:
                    output_parts.extend(parts)
    end = time.monotonic()
    result = {
        "ttft_s": round((first_token or end) - start, 4),
        "latency_s": round(end - start, 4),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens", 0),
        "request_id": request_id,
    }
    if capture_text:
        result["output_text"] = "".join(output_parts)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="qwen35-a3b-w8a8")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--target-output", action="store_true")
    parser.add_argument("--capture-text", action="store_true")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--global-concurrency", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = []
    open_input = gzip.open if args.input.suffix == ".gz" else Path.open
    with open_input(args.input, "rt", encoding="utf-8") as source:
        for line in source:
            if args.limit and len(rows) >= args.limit:
                break
            item = json.loads(line)
            rows.append(item)

    print_lock = threading.Lock()

    def run_one(item: dict, phase: str) -> dict:
        max_tokens = (
            item["metadata"]["output_tokens_target"]
            if args.target_output
            else args.max_tokens
        )
        result = complete(
            args.url,
            args.model,
            item["prompt"],
            max_tokens,
            args.capture_text,
            item["messages"] if args.chat else None,
        )
        record = {
            "id": item["id"],
            "group_id": item["group_id"],
            "phase": phase,
            "reuse_tier": item["metadata"]["reuse_tier"],
            "matched_flag": item["metadata"]["matched_flag"],
            "max_tokens": max_tokens,
            **result,
        }
        with print_lock:
            print(json.dumps(record, ensure_ascii=False), flush=True)
        return record

    group_seen: set[str] = set()
    results = []
    batch_start = time.monotonic()
    if args.global_concurrency:
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as executor:
            results = list(executor.map(lambda item: run_one(item, "mixed"), rows))
    elif args.concurrency == 1:
        for item in rows:
            group_id = item["group_id"]
            phase = "cold" if group_id not in group_seen else "warm"
            group_seen.add(group_id)
            results.append(run_one(item, phase))
    else:
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as executor:
            for offset in range(0, len(rows), 5):
                group = rows[offset : offset + 5]
                if not group:
                    continue
                group_id = group[0]["group_id"]
                if any(row["group_id"] != group_id for row in group):
                    raise ValueError("Concurrent replay requires intact 5-item groups")
                phase = "cold" if group_id not in group_seen else "warm"
                group_seen.add(group_id)
                results.append(run_one(group[0], phase))
                futures = [executor.submit(run_one, item, "warm") for item in group[1:]]
                results.extend(future.result() for future in futures)

    wall = time.monotonic() - batch_start
    summary = {
        "phase": "summary",
        "requests": len(results),
        "wall_s": round(wall, 4),
        "requests_per_s": round(len(results) / wall, 4),
        "output_tokens_per_s": round(
            sum(row["completion_tokens"] for row in results) / wall, 4
        ),
    }
    for phase in ("cold", "warm"):
        samples = [row["ttft_s"] for row in results if row["phase"] == phase]
        if samples:
            summary[f"{phase}_ttft_median_s"] = round(statistics.median(samples), 4)
            summary[f"{phase}_ttft_mean_s"] = round(statistics.mean(samples), 4)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
