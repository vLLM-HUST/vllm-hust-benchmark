#!/usr/bin/env python3
"""Send a request dataset to an OpenAI-compatible endpoint, each request at its
recorded arrival time, and record what happened to every request.

    python experiments/workload/replay_requests.py \\
        --workload data/workloads/gamma-cv2-zipf-medium \\
        --base-url http://127.0.0.1:8100/v1 --model qwen2.5-14b \\
        --out data/results/workload/<experiment>/<run>

It makes no decisions: the dataset says when and how much, this program only
sends. Reads `requests.jsonl` in Mooncake's format (`timestamp` ms,
`input_length` and `output_length` tokens), which is what
`generate_workload.py` writes.

Open loop, as in Llumnix's and vLLM's benchmarks: request i is sent at
start + timestamp_i whether or not earlier requests have finished, so a slow
server builds a queue instead of slowing the arrivals down.

How a length becomes a request (vLLM's /v1/completions):

- input:  `prompt` is a list of token ids, so the engine sees exactly
          `input_length` tokens with no tokenizer involved. The ids are drawn
          uniformly from [TOKEN_ID_MINIMUM, TOKEN_ID_MAXIMUM), per request from
          the stream "<seed>/prompt/<index>", so two runs send the same tokens.
          Random ids share no prefix: prefix caching has nothing to hit.
- output: `max_tokens = min_tokens = output_length` and `ignore_eos = true`, so
          the engine generates exactly `output_length` tokens whatever it says.
- both are checked afterwards against the engine's own `usage` counts; a
  mismatch is reported in the summary instead of silently skewing the numbers.

Writes into --out:
    requests.jsonl  one line per request: when it was due, sent, got its first
                    token, finished; token counts the engine reported; status
    summary.json    parameters, latency percentiles, throughput, mismatches,
                    the workload's manifest, and evidence_type
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from engine_metrics import EngineMetricsSampler

# Token ids to draw prompts from. 30000 is below the vocabulary size of the
# common model families (Llama 32000, OPT 50272, Qwen2.5 151643 regular
# tokens), and 1000 skips the low ids where tokenizers keep special tokens.
TOKEN_ID_MINIMUM = 1000
TOKEN_ID_MAXIMUM = 30000
PROGRESS_EVERY_SECONDS = 10.0


@dataclass
class RequestOutcome:
    index: int
    due_seconds: float  # when the dataset says it arrives, from the start
    sent_seconds: float  # when it was actually sent
    first_token_seconds: float | None
    finished_seconds: float | None
    dataset_input_length: int
    dataset_output_length: int
    engine_prompt_tokens: int | None  # from the engine's `usage`
    engine_completion_tokens: int | None
    status: str  # ok | error
    error: str = ""


def load_requests(workload: Path) -> list[dict]:
    with (workload / "requests.jsonl").open() as lines:
        return [json.loads(line) for line in lines if line.strip()]


def prompt_token_ids(seed: int, index: int, length: int) -> list[int]:
    random_generator = random.Random(f"{seed}/prompt/{index}")
    return [
        random_generator.randrange(TOKEN_ID_MINIMUM, TOKEN_ID_MAXIMUM)
        for _ in range(length)
    ]


def send_one(
    arguments: argparse.Namespace, index: int, request: dict, start: float
) -> RequestOutcome:
    """Send one request, streamed, and time it. Blocks until the stream ends."""
    body = {
        "model": arguments.model,
        "prompt": prompt_token_ids(arguments.seed, index, request["input_length"]),
        "max_tokens": request["output_length"],
        "min_tokens": request["output_length"],
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    http_request = urllib.request.Request(
        f"{arguments.base_url.rstrip('/')}/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    sent = time.monotonic() - start
    first_token = None
    usage: dict = {}
    error = ""
    try:
        with urllib.request.urlopen(http_request, timeout=arguments.timeout) as reply:
            for raw_line in reply:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if first_token is None and choices and choices[0].get("text"):
                    first_token = time.monotonic() - start
    except urllib.error.HTTPError as http_error:
        detail = http_error.read()[:300].decode(errors="replace")
        error = f"http {http_error.code}: {detail}"
    except Exception as other_error:  # noqa: BLE001 - one bad request is data
        error = f"{type(other_error).__name__}: {other_error}"
    return RequestOutcome(
        index=index,
        due_seconds=round(request["timestamp"] / 1000, 3),
        sent_seconds=round(sent, 4),
        first_token_seconds=None if first_token is None else round(first_token, 4),
        finished_seconds=round(time.monotonic() - start, 4),
        dataset_input_length=request["input_length"],
        dataset_output_length=request["output_length"],
        engine_prompt_tokens=usage.get("prompt_tokens"),
        engine_completion_tokens=usage.get("completion_tokens"),
        status="error" if error else "ok",
        error=error,
    )


def percentiles(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 4)

    return {
        "mean": round(statistics.fmean(ordered), 4),
        "p50": at(0.50),
        "p90": at(0.90),
        "p99": at(0.99),
        "maximum": round(ordered[-1], 4),
    }


def summarise(outcomes: list[RequestOutcome]) -> dict:
    """Latency definitions are vLLM's (`vllm bench serve`):
    TTFT = first token - sent; TPOT = (finished - first token) / (output - 1);
    end-to-end latency = finished - sent."""
    ok = [outcome for outcome in outcomes if outcome.status == "ok"]
    time_to_first_token = [o.first_token_seconds - o.sent_seconds for o in ok]
    time_per_output_token = [
        (o.finished_seconds - o.first_token_seconds) / (o.engine_completion_tokens - 1)
        for o in ok
        if o.engine_completion_tokens and o.engine_completion_tokens > 1
    ]
    end_to_end = [o.finished_seconds - o.sent_seconds for o in ok]
    send_lag = [o.sent_seconds - o.due_seconds for o in outcomes]
    span = max(o.finished_seconds for o in outcomes) if outcomes else 0.0
    input_mismatch = [
        o.index for o in ok if o.engine_prompt_tokens != o.dataset_input_length
    ]
    output_mismatch = [
        o.index for o in ok if o.engine_completion_tokens != o.dataset_output_length
    ]
    return {
        "requests": len(outcomes),
        "succeeded": len(ok),
        "failed": len(outcomes) - len(ok),
        "first_errors": [o.error for o in outcomes if o.error][:5],
        "duration_seconds": round(span, 3),
        "time_to_first_token_seconds": percentiles(time_to_first_token),
        "time_per_output_token_seconds": percentiles(time_per_output_token),
        "end_to_end_latency_seconds": percentiles(end_to_end),
        "send_lag_seconds": percentiles(send_lag),
        "input_tokens_per_second": round(
            sum(o.engine_prompt_tokens or 0 for o in ok) / span, 1
        )
        if span
        else 0.0,
        "output_tokens_per_second": round(
            sum(o.engine_completion_tokens or 0 for o in ok) / span, 1
        )
        if span
        else 0.0,
        "input_length_mismatches": {
            "count": len(input_mismatch),
            "first_indices": input_mismatch[:10],
        },
        "output_length_mismatches": {
            "count": len(output_mismatch),
            "first_indices": output_mismatch[:10],
        },
    }


def run(arguments: argparse.Namespace) -> list[RequestOutcome]:
    requests = load_requests(arguments.workload)
    if arguments.limit:
        requests = requests[: arguments.limit]
    outcomes: list[RequestOutcome] = []
    lock = threading.Lock()
    threads: list[threading.Thread] = []
    start = time.monotonic()

    def worker(index: int, request: dict) -> None:
        outcome = send_one(arguments, index, request, start)
        with lock:
            outcomes.append(outcome)

    next_progress = PROGRESS_EVERY_SECONDS
    sampler = EngineMetricsSampler(
        [arguments.base_url], arguments.out / "metrics.jsonl", start
    )
    with sampler:
        for index, request in enumerate(requests):
            due = request["timestamp"] / 1000
            while (now := time.monotonic() - start) < due:
                if now >= next_progress:
                    with lock:
                        finished = len(outcomes)
                    print(
                        f"[replay] t={now:7.1f}s sent={index} finished={finished} "
                        f"in_flight={index - finished}",
                        flush=True,
                    )
                    next_progress += PROGRESS_EVERY_SECONDS
                time.sleep(min(due - now, 0.05))
            thread = threading.Thread(target=worker, args=(index, request), daemon=True)
            thread.start()
            threads.append(thread)
        print(f"[replay] all {len(requests)} sent; waiting for the rest", flush=True)
        for thread in threads:
            thread.join()
    print(
        f"[replay] engine metrics: {sampler.samples} samples,"
        f" {sampler.failures} failed reads",
        flush=True,
    )
    return sorted(outcomes, key=lambda outcome: outcome.index)


def write(arguments: argparse.Namespace, outcomes: list[RequestOutcome]) -> dict:
    arguments.out.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps(asdict(o), separators=(",", ":")) + "\n" for o in outcomes
    )
    (arguments.out / "requests.jsonl").write_text(lines)
    manifest_path = arguments.workload / "manifest.json"
    revision_path = Path(__file__).resolve().parents[2] / "REVISION"
    summary = {
        "evidence_type": "live",
        "what_this_is": (
            "a synthetic request dataset sent open-loop to a real engine; "
            "the workload is generated, the latencies are measured"
        ),
        "parameters": {
            "workload": str(arguments.workload),
            "workload_requests_sha256": hashlib.sha256(
                (arguments.workload / "requests.jsonl").read_bytes()
            ).hexdigest(),
            "base_url": arguments.base_url,
            "model": arguments.model,
            "seed": arguments.seed,
            "limit": arguments.limit,
            "token_id_range": [TOKEN_ID_MINIMUM, TOKEN_ID_MAXIMUM],
        },
        "revision": json.loads(revision_path.read_text())
        if revision_path.exists()
        else None,
        "result": summarise(outcomes),
        "workload_manifest": json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else None,
    }
    (arguments.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiments/workload/replay_requests.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workload", type=Path, required=True, help="a directory")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", required=True, help="the served model name")
    parser.add_argument("--out", type=Path, required=True, help="a directory")
    parser.add_argument("--seed", type=int, default=11, help="for the prompt tokens")
    parser.add_argument("--limit", type=int, default=0, help="first N requests only")
    parser.add_argument("--timeout", type=float, default=1800.0, help="per request")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    outcomes = run(arguments)
    result = write(arguments, outcomes)["result"]
    print(json.dumps(result, indent=1), flush=True)
    print(f"[replay] wrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
