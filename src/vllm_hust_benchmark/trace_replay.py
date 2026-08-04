from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import heapq
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
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
    effective_prefix_tokens: int = 0
    token_segments: tuple[tuple[int, int, int], ...] = ()


def _seed_offset(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile_value / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _token_distribution(values: list[int]) -> dict[str, int | float]:
    numeric = [float(value) for value in values]
    return {
        "count": len(values),
        "min": min(values),
        "p50": _percentile(numeric, 50) or 0.0,
        "p95": _percentile(numeric, 95) or 0.0,
        "p99": _percentile(numeric, 99) or 0.0,
        "max": max(values),
        "total": sum(values),
    }


def _offset_at(segments: tuple[tuple[int, int, int], ...], position: int) -> int | None:
    for start, end, offset in segments:
        if start <= position < end:
            return offset
    return None


def _clip_segments(
    segments: tuple[tuple[int, int, int], ...], length: int
) -> list[tuple[int, int, int]]:
    return [
        (start, min(end, length), offset)
        for start, end, offset in segments
        if start < length
    ]


def _plan_token_segments(
    request: TraceRequest,
    input_tokens: int,
    previous: tuple[tuple[int, int, int], ...] | None,
) -> tuple[int, tuple[tuple[int, int, int], ...]]:
    """Build a compact, deterministic token tape for one replayed request.

    TraceLab's reported prefix is copied from the preceding replayed prompt in
    the same session, up to the amount of history actually available.  New
    append positions receive a different offset, which makes the first token
    after the effective prefix differ deterministically.  BurstGPT does not
    publish prefix overlap, so its entire payload is request-specific.
    """

    previous = previous or ()
    previous_length = max((end for _, end, _ in previous), default=0)
    reported_prefix = request.prefix_tokens if request.prefix_tokens is not None else 0
    effective_prefix = min(reported_prefix, input_tokens, previous_length)
    segments = _clip_segments(previous, effective_prefix)
    if effective_prefix < input_tokens:
        previous_offset = _offset_at(previous, effective_prefix)
        new_offset = (
            previous_offset + 1
            if previous_offset is not None
            else _seed_offset(request.request_id)
        )
        segments.append((effective_prefix, input_tokens, new_offset))
    return effective_prefix, tuple(segments)


def build_prompt_token_ids(
    item: PlannedRequest,
    *,
    token_id_min: int = 1000,
    token_id_max: int = 30000,
) -> list[int]:
    """Materialize the deterministic token tape recorded in a replay plan."""

    if not item.token_segments:
        return deterministic_token_ids(
            item.request,
            input_tokens=item.replay_input_tokens,
            token_id_min=token_id_min,
            token_id_max=token_id_max,
        )
    if token_id_min < 0 or token_id_max <= token_id_min:
        raise ValueError("invalid synthetic token ID range")
    width = token_id_max - token_id_min + 1
    tokens = [0] * item.replay_input_tokens
    for start, end, offset in item.token_segments:
        for position in range(start, end):
            tokens[position] = token_id_min + (offset + position * 104729) % width
    return tokens


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
    exclusion_stats: dict[str, int] | None = None,
) -> tuple[list[PlannedRequest], int]:
    if max_requests <= 0:
        raise ValueError("max_requests must be positive")
    if max_model_len <= 0:
        raise ValueError("max_model_len must be positive")
    if time_scale <= 0:
        raise ValueError("time_scale must be positive")

    failure_keys: list[tuple[float, str]] = []
    overflow_requests: list[TraceRequest] = []

    def replayable_requests() -> Any:
        for request in iter_trace_requests(target_id, trace_path):
            if provider and request.provider != provider:
                continue
            key = (request.arrival_s, request.request_id)
            if request.output_tokens == 0:
                failure_keys.append(key)
                continue
            replay_input_tokens = request.input_tokens
            if request.input_tokens + request.output_tokens > max_model_len:
                if overflow_policy in {"reject", "exclude-overflow"}:
                    overflow_requests.append(request)
                    continue
                if overflow_policy != "truncate-input":
                    raise ValueError(f"unsupported overflow policy: {overflow_policy}")
                replay_input_tokens = max_model_len - request.output_tokens
                if replay_input_tokens <= 0:
                    raise ValueError(
                        f"request {request.request_id} output_tokens="
                        f"{request.output_tokens} leaves no prompt capacity"
                    )
            yield key, request, replay_input_tokens

    selected = heapq.nsmallest(
        max_requests, replayable_requests(), key=lambda item: item[0]
    )
    requests = [(request, input_tokens) for _, request, input_tokens in selected]
    cutoff = selected[-1][0] if len(selected) == max_requests else (float("inf"), "")
    source_failures = sum(key <= cutoff for key in failure_keys)
    cohort_overflows = [
        request
        for request in overflow_requests
        if (request.arrival_s, request.request_id) <= cutoff
    ]
    overflow_rows = len(cohort_overflows)
    if overflow_policy == "reject" and cohort_overflows:
        request = min(
            cohort_overflows, key=lambda item: (item.arrival_s, item.request_id)
        )
        raise ValueError(
            f"request {request.request_id} needs "
            f"{request.input_tokens + request.output_tokens} tokens, exceeding "
            f"max_model_len={max_model_len}"
        )
    if not requests:
        raise ValueError("no replayable requests selected from trace")

    if exclusion_stats is not None:
        exclusion_stats.update(
            {
                "source_failure_rows_skipped": source_failures,
                "context_overflow_rows_excluded": overflow_rows,
            }
        )

    elapsed = 0.0
    previous_arrival = requests[0][0].arrival_s
    plan: list[PlannedRequest] = []
    session_tapes: dict[str, tuple[tuple[int, int, int], ...]] = {}
    for request, input_tokens in requests:
        delta = max(0.0, request.arrival_s - previous_arrival) / time_scale
        if max_interarrival_s is not None:
            if max_interarrival_s < 0:
                raise ValueError("max_interarrival_s must be non-negative")
            delta = min(delta, max_interarrival_s)
        elapsed += delta
        previous_arrival = request.arrival_s

        effective_prefix, token_segments = _plan_token_segments(
            request, input_tokens, session_tapes.get(request.session_id)
        )
        session_tapes[request.session_id] = token_segments
        plan.append(
            PlannedRequest(
                request=request,
                scheduled_offset_s=elapsed,
                replay_input_tokens=input_tokens,
                effective_prefix_tokens=effective_prefix,
                token_segments=token_segments,
            )
        )
    return plan, source_failures


def summarize_plan(
    target_id: str,
    plan: list[PlannedRequest],
    *,
    source_failures: int,
    context_overflow_rows_excluded: int = 0,
    trace_asset_sha256: str | None = None,
    max_requests: int | None = None,
    max_model_len: int | None = None,
    overflow_policy: str | None = None,
    time_scale: float | None = None,
    max_interarrival_s: float | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    inputs = [item.replay_input_tokens for item in plan]
    outputs = [item.request.output_tokens for item in plan]
    sessions = {item.request.session_id for item in plan}
    triggers: dict[str, int] = {}
    for item in plan:
        trigger = item.request.trigger or "unknown"
        triggers[trigger] = triggers.get(trigger, 0) + 1
    selected_requests = [
        {
            "request_id": item.request.request_id,
            "arrival_s": item.request.arrival_s,
            "input_tokens": item.replay_input_tokens,
            "prefix_tokens": item.request.prefix_tokens,
            "append_tokens": item.request.append_tokens,
            "effective_prefix_tokens": item.effective_prefix_tokens,
            "output_tokens": item.request.output_tokens,
        }
        for item in plan
    ]
    selected_requests_sha256 = _canonical_sha256(selected_requests)
    signature_payload = {
        "target_id": target_id,
        "trace_asset_sha256": trace_asset_sha256,
        "max_requests": max_requests if max_requests is not None else len(plan),
        "max_model_len": max_model_len,
        "overflow_policy": overflow_policy,
        "time_scale": time_scale,
        "max_interarrival_s": max_interarrival_s,
        "provider": provider,
        "selected_requests_sha256": selected_requests_sha256,
    }
    cohort_setting_signature = _canonical_sha256(signature_payload)
    return {
        "schema_version": "official-trace-replay-plan/v1",
        "target_id": target_id,
        "requests": len(plan),
        "sessions": len(sessions),
        "source_failure_rows_skipped": source_failures,
        "context_overflow_rows_excluded": context_overflow_rows_excluded,
        "scheduled_duration_s": plan[-1].scheduled_offset_s,
        "input_tokens": _token_distribution(inputs),
        "output_tokens": _token_distribution(outputs),
        "selected_requests_sha256": selected_requests_sha256,
        "cohort_setting_signature": cohort_setting_signature,
        "cohort": {
            "selected_requests_sha256": selected_requests_sha256,
            "setting_signature_payload": signature_payload,
            "cohort_setting_signature": cohort_setting_signature,
        },
        "triggers": triggers,
    }


def summarize_results(
    plan: list[PlannedRequest], results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build a vLLM-bench-compatible summary without relabelling E2E latency as TTFT."""

    completed = sum(
        result.get("http_status") == 200 and "error" not in result for result in results
    )
    failed = len(results) - completed
    duration_s = max(
        (
            float(
                result.get("finished_offset_s")
                if result.get("finished_offset_s") is not None
                else float(result.get("scheduled_offset_s") or 0.0)
                + float(result.get("replay_latency_s") or 0.0)
            )
            for result in results
        ),
        default=0.0,
    )
    latencies = [
        float(result.get("e2e_latency_s") or result["replay_latency_s"])
        for result in results
        if result.get("http_status") == 200 and "error" not in result
    ]
    successful = [
        result
        for result in results
        if result.get("http_status") == 200 and "error" not in result
    ]
    input_tokens = sum(
        int(result["response_usage"]["prompt_tokens"]) for result in successful
    )
    output_tokens = sum(
        int(result["response_usage"]["completion_tokens"]) for result in successful
    )

    return {
        "completed": completed,
        "failed": failed,
        "duration": duration_s,
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "request_throughput": completed / duration_s if duration_s else 0.0,
        "output_throughput": output_tokens / duration_s if duration_s else 0.0,
        "mean_e2e_latency_ms": mean(latencies) * 1000.0 if latencies else None,
        "p50_e2e_latency_ms": (
            _percentile(latencies, 50) * 1000.0 if latencies else None
        ),
        "p95_e2e_latency_ms": (
            _percentile(latencies, 95) * 1000.0 if latencies else None
        ),
        "p99_e2e_latency_ms": (
            _percentile(latencies, 99) * 1000.0 if latencies else None
        ),
        "errors": [
            result.get("error") or result.get("http_status")
            for result in results
            if result.get("http_status") != 200 or "error" in result
        ],
    }


def summarize_executed_inputs(
    plan: list[PlannedRequest], results: list[dict[str, Any]]
) -> dict[str, str]:
    """Hash the exact token-ID payloads materialized immediately before HTTP send."""
    if len(results) != len(plan):
        raise ValueError("executed input evidence count does not match replay plan")
    executed: list[dict[str, str]] = []
    for item, result in zip(plan, results, strict=True):
        request = result.get("request")
        request_id = request.get("request_id") if isinstance(request, dict) else None
        if request_id != item.request.request_id:
            raise ValueError("executed input evidence order does not match replay plan")
        token_digest = result.get("prompt_token_ids_sha256")
        payload_digest = result.get("request_payload_sha256")
        if not all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for digest in (token_digest, payload_digest)
        ):
            raise ValueError("replay result lacks exact prompt token-ID evidence")
        executed.append(
            {
                "request_id": item.request.request_id,
                "prompt_token_ids_sha256": token_digest,
                "request_payload_sha256": payload_digest,
            }
        )
    return {
        "resolved_input_kind": "production-trace-prompt-token-ids",
        "resolved_input_sha256": _canonical_sha256(executed),
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
    replay_started: float,
) -> dict[str, Any]:
    request = item.request
    actual_started = time.monotonic()
    prompt_token_ids = build_prompt_token_ids(
        item,
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
    wire_payload_bytes = http_request.data
    if not isinstance(wire_payload_bytes, bytes):
        raise TypeError("completion request did not retain an exact byte payload")
    wire_payload = json.loads(wire_payload_bytes)
    wire_prompt_token_ids = wire_payload.get("prompt")
    if wire_prompt_token_ids != prompt_token_ids:
        raise RuntimeError("completion request prompt drifted before HTTP send")
    prompt_token_ids_sha256 = _canonical_sha256(wire_prompt_token_ids)
    request_payload_sha256 = hashlib.sha256(wire_payload_bytes).hexdigest()
    started = time.monotonic()
    try:
        with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
            body = response.read()
            decoded = json.loads(body)
            finished = time.monotonic()
            usage = decoded.get("usage")
            if not isinstance(usage, dict) or not all(
                isinstance(usage.get(key), int) and usage[key] >= 0
                for key in ("prompt_tokens", "completion_tokens")
            ):
                raise ValueError("successful completion response has invalid usage")
            if usage["prompt_tokens"] != item.replay_input_tokens:
                raise ValueError(
                    "successful completion response prompt_tokens does not match "
                    "the replay payload"
                )
            if usage["completion_tokens"] != request.output_tokens:
                raise ValueError(
                    "successful completion response completion_tokens does not "
                    "match the requested exact output length"
                )
            return {
                "request": request.to_dict(),
                "scheduled_offset_s": item.scheduled_offset_s,
                "replay_input_tokens": item.replay_input_tokens,
                "effective_prefix_tokens": item.effective_prefix_tokens,
                "http_status": response.status,
                "actual_start_offset_s": actual_started - replay_started,
                "finished_offset_s": finished - replay_started,
                "queue_delay_s": max(
                    0.0,
                    actual_started - replay_started - item.scheduled_offset_s,
                ),
                "replay_latency_s": finished - started,
                "e2e_latency_s": finished - (replay_started + item.scheduled_offset_s),
                "response_usage": usage,
                "response_sha256": hashlib.sha256(body).hexdigest(),
                "prompt_token_ids_sha256": prompt_token_ids_sha256,
                "request_payload_sha256": request_payload_sha256,
            }
    except urllib.error.HTTPError as exc:
        body = exc.read()
        finished = time.monotonic()
        return {
            "request": request.to_dict(),
            "scheduled_offset_s": item.scheduled_offset_s,
            "replay_input_tokens": item.replay_input_tokens,
            "effective_prefix_tokens": item.effective_prefix_tokens,
            "http_status": exc.code,
            "actual_start_offset_s": actual_started - replay_started,
            "finished_offset_s": finished - replay_started,
            "queue_delay_s": max(
                0.0, actual_started - replay_started - item.scheduled_offset_s
            ),
            "replay_latency_s": finished - started,
            "e2e_latency_s": finished - (replay_started + item.scheduled_offset_s),
            "error": body.decode("utf-8", errors="replace")[:2000],
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "request_payload_sha256": request_payload_sha256,
        }
    except Exception as exc:
        finished = time.monotonic()
        return {
            "request": request.to_dict(),
            "scheduled_offset_s": item.scheduled_offset_s,
            "replay_input_tokens": item.replay_input_tokens,
            "effective_prefix_tokens": item.effective_prefix_tokens,
            "actual_start_offset_s": actual_started - replay_started,
            "finished_offset_s": finished - replay_started,
            "queue_delay_s": max(
                0.0, actual_started - replay_started - item.scheduled_offset_s
            ),
            "replay_latency_s": finished - started,
            "e2e_latency_s": finished - (replay_started + item.scheduled_offset_s),
            "error": repr(exc),
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "request_payload_sha256": request_payload_sha256,
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
                    replay_started=started,
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
        "--overflow-policy",
        choices=["reject", "truncate-input", "exclude-overflow"],
        default="reject",
    )
    replay.add_argument("--time-scale", type=float, default=1.0)
    replay.add_argument("--max-interarrival-s", type=float)
    replay.add_argument("--provider", choices=["claude", "codex"])
    replay.add_argument("--token-id-min", type=int, default=1000)
    replay.add_argument("--token-id-max", type=int, default=30000)
    replay.add_argument("--timeout-s", type=float, default=600.0)
    replay.add_argument("--output", type=Path)
    replay.add_argument("--summary-output", type=Path)
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
    exclusion_stats: dict[str, int] = {}
    plan, source_failures = load_replay_plan(
        args.target_id,
        trace_path,
        max_requests=args.max_requests,
        max_model_len=args.max_model_len,
        overflow_policy=args.overflow_policy,
        time_scale=args.time_scale,
        max_interarrival_s=args.max_interarrival_s,
        provider=args.provider,
        exclusion_stats=exclusion_stats,
    )
    summary = summarize_plan(
        args.target_id,
        plan,
        source_failures=source_failures,
        context_overflow_rows_excluded=exclusion_stats.get(
            "context_overflow_rows_excluded", 0
        ),
        trace_asset_sha256=_sha256_file(trace_path),
        max_requests=args.max_requests,
        max_model_len=args.max_model_len,
        overflow_policy=args.overflow_policy,
        time_scale=args.time_scale,
        max_interarrival_s=args.max_interarrival_s,
        provider=args.provider,
    )
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
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        result_summary = summarize_results(plan, results)
        result_summary.update(summarize_executed_inputs(plan, results))
        result_summary.update(
            {
                "trace_plan": summary,
                "cohort_setting_signature": summary["cohort_setting_signature"],
                "selected_requests_sha256": summary["selected_requests_sha256"],
                "input_tokens": summary["input_tokens"],
                "output_tokens": summary["output_tokens"],
            }
        )
        args.summary_output.write_text(
            json.dumps(result_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
