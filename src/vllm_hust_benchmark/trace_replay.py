from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vllm_hust_benchmark.official_trace_targets import (
    TraceRequest,
    deterministic_token_ids,
    fetch_trace_asset,
    get_trace_target,
    iter_trace_requests,
    load_trace_target_registry,
    verify_trace_asset,
)


@dataclass(frozen=True)
class PlannedRequest:
    request: TraceRequest
    scheduled_offset_s: float
    replay_input_tokens: int


def load_replay_plan(
    target_id: str,
    trace_path: Path,
    *,
    max_requests: int,
    max_model_len: int,
    overflow_policy: str,
    time_scale: float,
    max_interarrival_s: float | None,
    provider: str | None = None,
) -> tuple[list[PlannedRequest], int]:
    if max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")

    requests: list[TraceRequest] = []
    source_failures = 0
    for request in iter_trace_requests(target_id, trace_path):
        if provider and request.provider != provider:
            continue
        if request.output_tokens == 0:
            source_failures += 1
            continue
        requests.append(request)
        if len(requests) >= max_requests:
            break
    requests.sort(key=lambda item: (item.arrival_s, item.request_id))
    if not requests:
        raise ValueError("no replayable requests selected from trace")

    elapsed = 0.0
    previous_arrival = requests[0].arrival_s
    plan: list[PlannedRequest] = []
    for request in requests:
        delta = max(0.0, request.arrival_s - previous_arrival) / time_scale
        if max_interarrival_s is not None:
            if max_interarrival_s < 0:
                raise ValueError("max_interarrival_s must be non-negative")
            delta = min(delta, max_interarrival_s)
        elapsed += delta
        previous_arrival = request.arrival_s

        input_tokens = request.input_tokens
        if input_tokens + request.output_tokens > max_model_len:
            if overflow_policy == "reject":
                raise ValueError(
                    f"request {request.request_id} needs "
                    f"{input_tokens + request.output_tokens} tokens, exceeding "
                    f"max_model_len={max_model_len}"
                )
            if overflow_policy != "truncate-input":
                raise ValueError(f"unsupported overflow policy: {overflow_policy}")
            input_tokens = max_model_len - request.output_tokens
            if input_tokens <= 0:
                raise ValueError(
                    f"request {request.request_id} output_tokens="
                    f"{request.output_tokens} leaves no prompt capacity"
                )
        plan.append(
            PlannedRequest(
                request=request,
                scheduled_offset_s=elapsed,
                replay_input_tokens=input_tokens,
            )
        )
    return plan, source_failures


def summarize_plan(
    target_id: str,
    plan: list[PlannedRequest],
    *,
    source_failures: int,
) -> dict[str, Any]:
    inputs = [item.replay_input_tokens for item in plan]
    outputs = [item.request.output_tokens for item in plan]
    sessions = {item.request.session_id for item in plan}
    triggers: dict[str, int] = {}
    for item in plan:
        trigger = item.request.trigger or "unknown"
        triggers[trigger] = triggers.get(trigger, 0) + 1
    return {
        "schema_version": "official-trace-replay-plan/v1",
        "target_id": target_id,
        "requests": len(plan),
        "sessions": len(sessions),
        "source_failure_rows_skipped": source_failures,
        "scheduled_duration_s": plan[-1].scheduled_offset_s,
        "input_tokens": {
            "min": min(inputs),
            "max": max(inputs),
            "total": sum(inputs),
        },
        "output_tokens": {
            "min": min(outputs),
            "max": max(outputs),
            "total": sum(outputs),
        },
        "triggers": triggers,
    }


def _invoke(
    item: PlannedRequest,
    *,
    url: str,
    model: str,
    api_key: str | None,
    timeout_s: float,
    token_id_min: int,
    token_id_max: int,
) -> dict[str, Any]:
    request = item.request
    prompt_token_ids = deterministic_token_ids(
        request,
        input_tokens=item.replay_input_tokens,
        token_id_min=token_id_min,
        token_id_max=token_id_max,
    )
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt_token_ids,
            "max_tokens": request.output_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    http_request = urllib.request.Request(url, data=payload, headers=headers)
    started = time.monotonic()
    try:
        with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
            body = response.read()
            decoded = json.loads(body)
            return {
                "request": request.to_dict(),
                "scheduled_offset_s": item.scheduled_offset_s,
                "replay_input_tokens": item.replay_input_tokens,
                "http_status": response.status,
                "replay_latency_s": time.monotonic() - started,
                "response_usage": decoded.get("usage"),
                "response_sha256": hashlib.sha256(body).hexdigest(),
            }
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return {
            "request": request.to_dict(),
            "scheduled_offset_s": item.scheduled_offset_s,
            "replay_input_tokens": item.replay_input_tokens,
            "http_status": exc.code,
            "replay_latency_s": time.monotonic() - started,
            "error": body.decode("utf-8", errors="replace")[:2000],
        }
    except Exception as exc:
        return {
            "request": request.to_dict(),
            "scheduled_offset_s": item.scheduled_offset_s,
            "replay_input_tokens": item.replay_input_tokens,
            "replay_latency_s": time.monotonic() - started,
            "error": repr(exc),
        }


def execute_replay(
    plan: list[PlannedRequest],
    *,
    base_url: str,
    endpoint: str,
    model: str,
    api_key: str | None,
    max_concurrency: int,
    timeout_s: float,
    token_id_min: int,
    token_id_max: int,
) -> list[dict[str, Any]]:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive")
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    started = time.monotonic()
    futures: list[concurrent.futures.Future[dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        for item in plan:
            delay = started + item.scheduled_offset_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            futures.append(
                pool.submit(
                    _invoke,
                    item,
                    url=url,
                    model=model,
                    api_key=api_key,
                    timeout_s=timeout_s,
                    token_id_min=token_id_min,
                    token_id_max=token_id_max,
                )
            )
        return [future.result() for future in futures]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="official-trace-targets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")

    show = subparsers.add_parser("show")
    show.add_argument("target_id")

    fetch = subparsers.add_parser("fetch")
    fetch.add_argument("target_id")
    fetch.add_argument("--cache-dir", type=Path, required=True)

    replay = subparsers.add_parser("replay")
    replay.add_argument("target_id")
    replay.add_argument("--trace-path", type=Path)
    replay.add_argument("--cache-dir", type=Path)
    replay.add_argument("--model", required=True)
    replay.add_argument("--base-url", default="http://127.0.0.1:8000")
    replay.add_argument("--endpoint", default="/v1/completions")
    replay.add_argument("--api-key-env", default="VLLM_HUST_API_KEY")
    replay.add_argument("--max-requests", type=int, default=1000)
    replay.add_argument("--max-concurrency", type=int, default=64)
    replay.add_argument("--max-model-len", type=int, required=True)
    replay.add_argument(
        "--overflow-policy", choices=["reject", "truncate-input"], default="reject"
    )
    replay.add_argument("--time-scale", type=float, default=1.0)
    replay.add_argument("--max-interarrival-s", type=float)
    replay.add_argument("--provider", choices=["claude", "codex"])
    replay.add_argument("--token-id-min", type=int, default=1000)
    replay.add_argument("--token-id-max", type=int, default=30000)
    replay.add_argument("--timeout-s", type=float, default=600.0)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "list":
        for target in load_trace_target_registry()["targets"]:
            print(f"{target['id']}\t{target['status']}\t{target['title']}")
        return 0
    if args.command == "show":
        print(
            json.dumps(get_trace_target(args.target_id), ensure_ascii=False, indent=2)
        )
        return 0
    if args.command == "fetch":
        print(fetch_trace_asset(args.target_id, args.cache_dir))
        return 0

    if args.trace_path is None:
        if args.cache_dir is None:
            raise ValueError("replay requires --trace-path or --cache-dir")
        trace_path = fetch_trace_asset(args.target_id, args.cache_dir)
    else:
        trace_path = args.trace_path
        verify_trace_asset(get_trace_target(args.target_id), trace_path)
    plan, source_failures = load_replay_plan(
        args.target_id,
        trace_path,
        max_requests=args.max_requests,
        max_model_len=args.max_model_len,
        overflow_policy=args.overflow_policy,
        time_scale=args.time_scale,
        max_interarrival_s=args.max_interarrival_s,
        provider=args.provider,
    )
    summary = summarize_plan(args.target_id, plan, source_failures=source_failures)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    if args.output is None:
        raise ValueError("non-dry replay requires --output")

    results = execute_replay(
        plan,
        base_url=args.base_url,
        endpoint=args.endpoint,
        model=args.model,
        api_key=os.environ.get(args.api_key_env),
        max_concurrency=args.max_concurrency,
        timeout_s=args.timeout_s,
        token_id_min=args.token_id_min,
        token_id_max=args.token_id_max,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "metadata",
                    "target": get_trace_target(args.target_id),
                    "plan": summary,
                    "model": args.model,
                    "base_url": args.base_url,
                    "endpoint": args.endpoint,
                    "overflow_policy": args.overflow_policy,
                    "time_scale": args.time_scale,
                    "max_interarrival_s": args.max_interarrival_s,
                    "synthetic_payload": True,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    failures = sum(
        1 for result in results if result.get("http_status") != 200 or "error" in result
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
