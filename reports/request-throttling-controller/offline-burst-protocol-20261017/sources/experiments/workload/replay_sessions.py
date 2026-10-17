#!/usr/bin/env python3
"""Replay a session dataset (agent sessions: rounds of LLM calls separated by
tool waits) against an OpenAI-compatible endpoint, and record every round.

    python experiments/workload/replay_sessions.py \\
        --workload data/workloads/tracelab-fit16k-min4rounds-n16 \\
        --base-url http://127.0.0.1:8100/v1 --model qwen2.5-14b \\
        --out data/results/workload/<experiment>/<run>

Reads `sessions.csv` in TraceLab's session_runner format (see
select_tracelab_sessions.py). It makes no decisions; the semantics are
TraceLab's runner (uw-syfi/TraceLab, replay/src), re-implemented in Python:

- sessions are open loop: session s starts at its `arrival_time` (ms) whether
  or not other sessions are done (session.rs, wait_for_session_arrival);
- rounds inside a session are closed loop: send round i, wait for the whole
  response, sleep `tool_wait_after_ms`, then send round i + 1
  (session.rs lines 106-107);
- the prompt of a round is the first `prefix_len` tokens of the session's
  context so far, followed by `input_len` new tokens; after the round the
  context becomes that prompt plus the model's actual output tokens, taken
  from vLLM's `return_token_ids` (tokens.rs, PromptBuilder). Carrying the real
  output forward keeps it prefix-cache hittable, as in a real agent. If the
  context is shorter than `prefix_len` (the trace's context was compacted or
  rebuilt), new tokens fill the gap, as TraceLab does;
- output is exactly `output_len` tokens: max_tokens = min_tokens = output_len,
  ignore_eos (backend.rs lines 75-83).

--duration-seconds (default: none) ends the run at a fixed time: no round is
sent after it, a tool wait that would run past it is cut short, and rounds
already in flight are allowed to finish. Programs not done by then are counted
as not completed. This is a measurement window, as in the earlier control-plane
experiments: the slowest recorded programs need well over an hour on one stream.

The one difference: TraceLab draws new tokens from a tokenized text corpus; here
they are random token ids, per session from the stream "<seed>/session/<id>".

Two ways to send (--api):
- completions (default): POST /v1/completions with `prompt` as the token-id
  list, ids from replay_requests.py's range. Exact lengths, no tokenizer needed.
  Talks to a bare vLLM engine.
- chat: POST /v1/chat/completions with one user message whose text is the
  token ids decoded by the model's tokenizer (--tokenizer). This is what a
  router that only speaks chat needs: ThunderAgent forwards to the engine's
  /v1/chat/completions only (ThunderAgent/backend/state.py, completions_url)
  and identifies a program by `program_id` in the body (app.py,
  get_program_id); when a program ends it is released with
  POST /programs/release (app.py). New tokens are drawn only from tokens that
  are one lowercase English word with a leading space (18,742 of them in
  Qwen2.5's vocabulary), so decoding and re-encoding gives back exactly the same
  ids (checked on 200 random texts of 50-3,000 tokens: 200 exact).
  ⚠️ The chat template adds a fixed number of tokens to every request (for
  Qwen2.5 a default system prompt, the same in every program, so it is a
  prefix shared across programs); the number is measured once and stored as
  `chat_template_tokens`, and the token-count check expects it. The model's own
  output, carried into the next round as text, may re-encode a few tokens
  differently; the check reports how often.

Writes into --out:
    rounds.jsonl   one line per round: session, round, when sent / first token /
                   finished, token counts planned and reported by the engine
    metrics.jsonl  every engine's counters once a second (engine_metrics.py);
                   engines are given with --engine-url (default: --base-url)
    summary.json   latency percentiles per round, session completion times,
                   mismatches, each engine's capacity (engine_capacity.py),
                   the workload's manifest, evidence_type
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from engine_capacity import measure as measure_engine_capacity
from engine_metrics import EngineMetricsSampler
from generate_workload import git_state
from raw_observation import DeviceSampler, EngineRawMetricsSampler
from replay_requests import TOKEN_ID_MAXIMUM, TOKEN_ID_MINIMUM, percentiles

PROGRESS_EVERY_SECONDS = 10.0


@dataclass
class RoundOutcome:
    session_id: str
    round_idx: int
    slot: int | None  # closed loop: the worker slot; open loop: None
    program_arrival_seconds: float | None  # external arrival; undefined in closed loop
    priority_class: str
    latency_objective_seconds: float | None
    sent_seconds: float
    first_token_seconds: float | None
    finished_seconds: float
    prefix_len: int  # planned, from the dataset
    input_len: int
    output_len: int
    tool_wait_after_ms: int
    engine_prompt_tokens: int | None
    engine_completion_tokens: int | None
    engine_cached_tokens: int | None  # vLLM's usage.prompt_tokens_details, if sent
    status: str
    error: str = ""
    prompt_token_ids_sha256: str = ""
    previous_history_token_count: int = 0
    previous_history_sha256: str = ""
    inherited_history_sha256: str = ""
    expected_engine_prompt_tokens: int | None = None
    rendered_prompt_token_ids_sha256: str = ""


def load_sessions(workload: Path) -> dict[str, list[dict]]:
    sessions: dict[str, list[dict]] = defaultdict(list)
    with (workload / "sessions.csv").open() as source:
        for row in csv.DictReader(source):
            sessions[row["session_id"]].append({k: row[k] for k in row})
    for rounds in sessions.values():
        rounds.sort(key=lambda round_: int(round_["round_idx"]))
    return dict(sessions)


class SessionPrompts:
    """TraceLab's PromptBuilder: the session's context, and the next prompt.

    New tokens come from `vocabulary` when given (chat mode: ids that survive
    decode and re-encode), otherwise from [TOKEN_ID_MINIMUM, TOKEN_ID_MAXIMUM)."""

    def __init__(self, seed: int, session_id: str, vocabulary: list[int] | None = None):
        self.random_generator = random.Random(f"{seed}/session/{session_id}")
        self.vocabulary = vocabulary
        self.context: list[int] = []

    def new_tokens(self, count: int) -> list[int]:
        if self.vocabulary:
            return [self.random_generator.choice(self.vocabulary) for _ in range(count)]
        return [
            self.random_generator.randrange(TOKEN_ID_MINIMUM, TOKEN_ID_MAXIMUM)
            for _ in range(count)
        ]

    def prompt(self, prefix_len: int, input_len: int) -> list[int]:
        if len(self.context) < prefix_len:
            self.context.extend(self.new_tokens(prefix_len - len(self.context)))
        return self.context[:prefix_len] + self.new_tokens(input_len)

    def commit(
        self, prompt: list[int], output: list[int], *, fixed_content: bool = False
    ) -> None:
        self.context = prompt + (
            self.new_tokens(len(output)) if fixed_content else output
        )


def stable_chat_vocabulary(tokenizer) -> list[int]:
    """Only use words whose individual token IDs survive decoding and encoding."""
    import re
    return sorted(token_id for token, token_id in tokenizer.get_vocab().items()
        if re.fullmatch(r" [a-z]{4,9}", tokenizer.convert_tokens_to_string([token]))
        and tokenizer.encode(tokenizer.decode([token_id]), add_special_tokens=False) == [token_id])


class ChatRendering:
    """Token ids <-> text for --api chat, with the model's tokenizer."""

    def __init__(self, tokenizer_path: str, chat_template_kwargs: dict | None = None):
        from transformers import AutoTokenizer  # only chat mode needs it

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.vocabulary = stable_chat_vocabulary(self.tokenizer)
        templated = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            tokenize=False,
            **self.chat_template_kwargs,
        )
        self.chat_template_tokens = len(
            self.tokenizer.encode(templated, add_special_tokens=False)
        )

    def text(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def rendered_prompt_token_ids(self, token_ids: list[int]) -> list[int]:
        text = self.text(token_ids)
        if self.tokenizer.encode(text, add_special_tokens=False) != token_ids:
            raise ValueError('Chat text does not preserve planned token IDs')
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True,
            tokenize=True, return_dict=False, **self.chat_template_kwargs)


def release_program(arguments: argparse.Namespace, program_id: str) -> None:
    """Tell a program-aware router the program is done; a bare engine answers
    404, which is fine."""
    root = arguments.base_url.rstrip("/").removesuffix("/v1")
    request = urllib.request.Request(
        f"{root}/programs/release",
        data=json.dumps({"program_id": program_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with contextlib.suppress(Exception):  # absent on a bare engine
        urllib.request.urlopen(request, timeout=10).read()


def send_round(
    arguments: argparse.Namespace,
    prompt: list[int],
    output_len: int,
    start: float,
    session_id: str,
    chat: ChatRendering | None,
) -> tuple[float, float | None, list[int], dict, str, list[list[float]]]:
    """Returns (sent, first token, output token ids, usage, error, arrivals):
    `arrivals` is one [seconds since start, tokens] per streamed chunk, the
    ground truth for per-token latency."""
    if chat:
        path = "/chat/completions"
        request_input = {
            "messages": [{"role": "user", "content": chat.text(prompt)}],
            "program_id": session_id,
        }
        if chat.chat_template_kwargs:
            request_input["chat_template_kwargs"] = chat.chat_template_kwargs
    else:
        path = "/completions"
        request_input = {"prompt": prompt}
    body = request_input | {
        "model": arguments.model,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        f"{arguments.base_url.rstrip('/')}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    sent = time.monotonic() - start
    first_token = None
    output_ids: list[int] = []
    arrivals: list[list[float]] = []  # [seconds since start, tokens in chunk]
    usage: dict = {}
    error = ""
    try:
        remaining = getattr(arguments, "drain_deadline_seconds", None)
        timeout = arguments.timeout if arguments.timeout > 0 else None
        if remaining is not None:
            budget = max(0.001, remaining - sent)
            timeout = min(timeout, budget) if timeout is not None else budget
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            for raw_line in reply:
                if remaining is not None and time.monotonic() - start >= remaining:
                    raise TimeoutError("drain deadline reached")
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    text = choice.get("text") or (choice.get("delta") or {}).get(
                        "content"
                    )
                    if first_token is None and (text or choice.get("token_ids")):
                        first_token = time.monotonic() - start
                    chunk_ids = choice.get("token_ids") or []
                    output_ids.extend(chunk_ids)
                    if chunk_ids or text:
                        arrivals.append([
                            round(time.monotonic() - start, 4),
                            len(chunk_ids) or 1,
                        ])  # fmt: skip
    except urllib.error.HTTPError as http_error:
        detail = http_error.read()[:300].decode(errors="replace")
        error = f"http {http_error.code}: {detail}"
    except Exception as other_error:  # noqa: BLE001 - one bad round is data
        error = f"{type(other_error).__name__}: {other_error}"
    return sent, first_token, output_ids, usage, error, arrivals


def run_session(
    arguments: argparse.Namespace,
    session_id: str,
    rounds: list[dict],
    start: float,
    record: callable,
    chat: ChatRendering | None,
    record_tokens: callable = lambda line: None,
    slot: int | None = None,
) -> None:
    """Run one program. With a slot (closed loop) it starts at once; without
    one (open loop) it waits for its recorded arrival time."""
    arrival = int(rounds[0]["arrival_time"]) / 1000 if slot is None else None
    priority_class = rounds[0].get("priority_class") or "ordinary"
    raw_objective = rounds[0].get("latency_objective_seconds")
    latency_objective_seconds = float(raw_objective) if raw_objective else None
    if arrival is not None:
        while (now := time.monotonic() - start) < arrival:
            time.sleep(min(arrival - now, 0.05))
    prompts = SessionPrompts(
        arguments.seed, session_id, chat.vocabulary if chat else None
    )
    deadline = (
        getattr(arguments, "drain_deadline_seconds", None)
        or arguments.duration_seconds
        or float("inf")
    )
    for round_ in rounds:
        if time.monotonic() - start >= deadline:
            break
        prefix_len = int(round_["prefix_len"])
        input_len = int(round_["input_len"])
        output_len = int(round_["output_len"])
        tool_wait_after_ms = int(float(round_["tool_wait_after_ms"]))
        previous_history = list(prompts.context)
        prompt = prompts.prompt(prefix_len, input_len)
        rendered = chat.rendered_prompt_token_ids(prompt) if chat else prompt
        sent, first_token, output_ids, usage, error, arrivals = send_round(
            arguments, prompt, output_len, start, session_id, chat
        )
        record_tokens({
            "session_id": session_id, "round_idx": int(round_["round_idx"]),
            "slot": slot, "sent": round(sent, 4), "chunks": arrivals,
        })  # fmt: skip
        details = usage.get("prompt_tokens_details") or {}
        record(
            RoundOutcome(
                session_id=session_id,
                round_idx=int(round_["round_idx"]),
                slot=slot,
                program_arrival_seconds=arrival,
                priority_class=priority_class,
                latency_objective_seconds=latency_objective_seconds,
                sent_seconds=round(sent, 4),
                first_token_seconds=None
                if first_token is None
                else round(first_token, 4),
                finished_seconds=round(time.monotonic() - start, 4),
                prefix_len=prefix_len,
                input_len=input_len,
                output_len=output_len,
                tool_wait_after_ms=tool_wait_after_ms,
                engine_prompt_tokens=usage.get("prompt_tokens"),
                engine_completion_tokens=usage.get("completion_tokens"),
                engine_cached_tokens=details.get("cached_tokens"),
                status="error" if error else "ok",
                error=error,
                previous_history_token_count=len(previous_history),
                previous_history_sha256=hashlib.sha256(json.dumps(previous_history).encode()).hexdigest(),
                inherited_history_sha256=hashlib.sha256(json.dumps(prompt[:len(previous_history)]).encode()).hexdigest(),
                prompt_token_ids_sha256=hashlib.sha256(
                    json.dumps(prompt).encode()
                ).hexdigest(),
                expected_engine_prompt_tokens=len(rendered),
                rendered_prompt_token_ids_sha256=hashlib.sha256(json.dumps(rendered).encode()).hexdigest(),
            )
        )
        if error:
            break  # a session cannot continue without its context
        prompts.commit(
            prompt,
            output_ids,
            fixed_content=getattr(arguments, "fixed_program_content", False),
        )
        remaining = deadline - (time.monotonic() - start)
        time.sleep(max(0.0, min(tool_wait_after_ms / 1000, remaining)))
    release_program(arguments, session_id)


def summarise(
    sessions: dict[str, list[dict]],
    outcomes: list[RoundOutcome],
    chat_template_tokens: int = 0,
) -> dict:
    ok = [o for o in outcomes if o.status == "ok"]
    by_session: dict[str, list[RoundOutcome]] = defaultdict(list)
    for outcome in outcomes:
        by_session[outcome.session_id].append(outcome)
    completed = [
        session_id
        for session_id, rounds in sessions.items()
        if len(by_session[session_id]) == len(rounds)
        and all(o.status == "ok" for o in by_session[session_id])
    ]
    session_time = [
        by_session[s][-1].finished_seconds - by_session[s][0].sent_seconds
        for s in completed
    ]
    externally_arrived = [
        session_id
        for session_id, items in by_session.items()
        if items and items[0].program_arrival_seconds is not None
    ]
    end_to_end_time = [
        by_session[s][-1].finished_seconds
        - float(by_session[s][0].program_arrival_seconds)
        for s in completed
        if by_session[s][0].program_arrival_seconds is not None
    ]
    queue_before_first_send = [
        by_session[s][0].sent_seconds - float(by_session[s][0].program_arrival_seconds)
        for s in externally_arrived
    ]
    priority_summary = {}
    for priority_class in sorted(
        {by_session[s][0].priority_class for s in externally_arrived}
    ):
        arrived = [
            s
            for s in externally_arrived
            if by_session[s][0].priority_class == priority_class
        ]
        completed_in_class = [s for s in arrived if s in completed]
        latencies = [
            by_session[s][-1].finished_seconds
            - float(by_session[s][0].program_arrival_seconds)
            for s in completed_in_class
        ]
        attained = [
            s
            for s, latency in zip(completed_in_class, latencies, strict=True)
            if by_session[s][0].latency_objective_seconds is not None
            and latency <= float(by_session[s][0].latency_objective_seconds)
        ]
        priority_summary[priority_class] = {
            "programs_arrived": len(arrived),
            "programs_completed": len(completed_in_class),
            "programs_attaining_latency_objective": len(attained),
            "end_to_end_latency_seconds": percentiles(latencies),
        }
    tool_time = [
        sum(o.tool_wait_after_ms for o in by_session[s][:-1]) / 1000 for s in completed
    ]
    mismatches = {
        "prompt_tokens": [
            f"{o.session_id}/{o.round_idx}"
            for o in ok
            if o.engine_prompt_tokens
            != (o.expected_engine_prompt_tokens if o.expected_engine_prompt_tokens is not None
                else o.prefix_len + o.input_len + chat_template_tokens)
        ],
        "completion_tokens": [
            f"{o.session_id}/{o.round_idx}"
            for o in ok
            if o.engine_completion_tokens != o.output_len
        ],
    }
    span = max((o.finished_seconds for o in outcomes), default=0.0)
    output_tokens = sum(o.engine_completion_tokens or 0 for o in ok)
    return {
        "makespan_seconds": (
            round(
                span
                - min(
                    (
                        float(by_session[s][0].program_arrival_seconds)
                        for s in completed
                        if by_session[s][0].program_arrival_seconds is not None
                    ),
                    default=0.0,
                ),
                4,
            )
            if len(completed) == len(sessions) and sessions
            else None
        ),
        "sessions": len(sessions),
        "sessions_completed": len(completed),
        "sessions_unfinished": len(sessions) - len(completed),
        "run_seconds": round(span, 1),
        "rounds_completed_per_minute": round(len(ok) / (span / 60), 2) if span else 0.0,
        "output_tokens_total": output_tokens,
        "output_tokens_per_second": round(output_tokens / span, 1) if span else 0.0,
        "rounds": sum(len(rounds) for rounds in sessions.values()),
        "rounds_succeeded": len(ok),
        "rounds_failed": len(outcomes) - len(ok),
        "first_errors": [o.error for o in outcomes if o.error][:5],
        "time_to_first_token_seconds": percentiles(
            [
                o.first_token_seconds - o.sent_seconds
                for o in ok
                if o.first_token_seconds is not None
            ]
        ),
        "round_latency_seconds": percentiles(
            [o.finished_seconds - o.sent_seconds for o in ok]
        ),
        "session_completion_seconds": percentiles(session_time),
        "program_end_to_end_latency_seconds": percentiles(end_to_end_time),
        "program_queue_before_first_send_seconds": percentiles(queue_before_first_send),
        "programs_by_priority": priority_summary,
        "session_tool_wait_seconds": percentiles(tool_time),
        "engine_cached_tokens_reported": any(
            o.engine_cached_tokens is not None for o in ok
        ),
        "mismatches": {
            key: {"count": len(value), "first": value[:10]}
            for key, value in mismatches.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--chat-template-kwargs", type=json.loads, default={},
                        help='Explicit template settings applied to tokenizer and server alike')
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--evidence-type", choices=["live", "replay", "smoke"], default="live")
    parser.add_argument("--timeout", type=float, default=1800.0, help="socket timeout seconds; 0 disables it (batch drain deadline remains)")
    parser.add_argument("--api", choices=["completions", "chat"], default="completions")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="stop sending new rounds after this many seconds (default: to the end)",
    )
    parser.add_argument(
        "--tokenizer", help="model directory or tokenizer path; --api chat needs it"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        help="closed loop: keep this many programs running, starting the next "
        "program (in arrival order) as soon as one ends; 0 = open loop by "
        "recorded arrival time",
    )
    parser.add_argument(
        "--npu-cards",
        type=int,
        nargs="*",
        default=[],
        help="physical NPU ids to sample with npu-smi every 5 s",
    )
    parser.add_argument(
        "--engine-url",
        action="append",
        help="an engine's /v1 URL to sample metrics and capacity from; repeat for "
        "several engines behind a router (default: --base-url)",
    )
    parser.add_argument(
        "--drain-timeout-seconds",
        type=float,
        help="open loop: maximum drain time after the manifest arrival window; "
        "socket reads and streams are bounded, with release/progress overhead",
    )
    parser.add_argument(
        "--fixed-program-content",
        action="store_true",
        help="Use deterministic synthetic output placeholders for subsequent prompts",
    )
    arguments = parser.parse_args()
    engine_urls = arguments.engine_url or [arguments.base_url]
    if not isinstance(arguments.chat_template_kwargs, dict):
        parser.error('--chat-template-kwargs must be a JSON object')
    chat = ChatRendering(arguments.tokenizer, arguments.chat_template_kwargs) if arguments.api == "chat" else None
    if chat:
        print(
            f"[sessions] chat: {len(chat.vocabulary)} single-word tokens,"
            f" chat template adds {chat.chat_template_tokens} tokens per request",
            flush=True,
        )
    engines = []
    for url in engine_urls:
        capacity = measure_engine_capacity(url)
        print(
            f"[sessions] engine {url}: max_model_len={capacity['max_model_len']}"
            f" kv_capacity_tokens={capacity['kv_capacity_tokens_device']}"
            f" (+{capacity['kv_capacity_tokens_host']} in host memory)",
            flush=True,
        )
        engines.append(capacity)

    sessions = load_sessions(arguments.workload)
    arguments.drain_deadline_seconds = None
    if arguments.drain_timeout_seconds is not None:
        if arguments.concurrency or arguments.duration_seconds is not None:
            parser.error("bounded drain requires open loop without --duration-seconds")
        if arguments.drain_timeout_seconds <= 0:
            parser.error("--drain-timeout-seconds must be positive")
        manifest = json.loads((arguments.workload / "manifest.json").read_text())
        window = manifest.get("knobs", {}).get("arrival_duration_seconds")
        if window is None:
            parser.error("bounded drain requires manifest arrival_duration_seconds")
        arguments.drain_deadline_seconds = (
            float(window) + arguments.drain_timeout_seconds
        )
    outcomes: list[RoundOutcome] = []
    lock = threading.Lock()

    def record(outcome: RoundOutcome) -> None:
        with lock:
            outcomes.append(outcome)
            with (arguments.out / "rounds-live.jsonl").open("a") as live:
                live.write(json.dumps(asdict(outcome)) + "\n")

    arguments.out.mkdir(parents=True, exist_ok=True)
    # Keep the complete input beside the results, independent of host paths.
    (arguments.out / "sessions.csv").write_bytes(
        (arguments.workload / "sessions.csv").read_bytes()
    )
    token_file = (arguments.out / "tokens.jsonl").open("w")

    def record_tokens(line: dict) -> None:
        with lock:
            token_file.write(json.dumps(line, separators=(",", ":")) + "\n")

    order = sorted(sessions, key=lambda s: int(sessions[s][0]["arrival_time"]))
    started: list[str] = []
    start = time.monotonic()
    started_at_wall = time.time()
    deadline = (
        getattr(arguments, "drain_deadline_seconds", None)
        or arguments.duration_seconds
        or float("inf")
    )

    def slot_worker(slot: int) -> None:
        """Closed loop: take the next program as soon as the last one ends."""
        while time.monotonic() - start < deadline:
            with lock:
                if not order:
                    return
                session_id = order.pop(0)
                started.append(session_id)
            run_session(arguments, session_id, sessions[session_id], start, record,
                        chat, record_tokens, slot)  # fmt: skip

    if arguments.concurrency:
        threads = [
            threading.Thread(target=slot_worker, args=(slot,), daemon=True)
            for slot in range(arguments.concurrency)
        ]
    else:
        started = list(sessions)
        threads = [
            threading.Thread(
                target=run_session,
                args=(
                    arguments,
                    session_id,
                    rounds,
                    start,
                    record,
                    chat,
                    record_tokens,
                ),
                daemon=True,
            )  # fmt: skip
            for session_id, rounds in sessions.items()
        ]
    samplers = [
        EngineMetricsSampler(engine_urls, arguments.out / "metrics.jsonl", start),
        EngineRawMetricsSampler(
            engine_urls, arguments.out / "engine_metrics_raw.jsonl.gz", start
        ),
    ]
    if arguments.npu_cards:
        samplers.append(
            DeviceSampler(arguments.npu_cards, arguments.out / "device.jsonl.gz", start)
        )
    with contextlib.ExitStack() as stack:
        for sampler in samplers:
            stack.enter_context(sampler)
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads):
            time.sleep(PROGRESS_EVERY_SECONDS)
            elapsed = time.monotonic() - start
            with lock:
                done = len(outcomes)
                if arguments.concurrency:
                    arrived = len(started)
                else:
                    arrived = sum(
                        int(rounds[0]["arrival_time"]) / 1000 <= elapsed
                        for rounds in sessions.values()
                    )
            alive = sum(thread.is_alive() for thread in threads)
            print(
                f"[sessions] t={elapsed:7.1f}s rounds={done}"
                f" programs_arrived={arrived} program_threads_alive={alive}",
                flush=True,
            )
    token_file.close()
    for sampler in samplers:
        print(
            f"[sessions] {type(sampler).__name__}: {sampler.samples} samples,"
            f" {sampler.failures} failed reads",
            flush=True,
        )
    sessions = {session_id: sessions[session_id] for session_id in started}

    outcomes.sort(key=lambda o: (o.session_id, o.round_idx))
    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "rounds.jsonl").write_text(
        "".join(json.dumps(asdict(o), separators=(",", ":")) + "\n" for o in outcomes)
    )
    manifest_path = arguments.workload / "manifest.json"
    revision_path = Path(__file__).resolve().parents[2] / "REVISION"
    result = summarise(sessions, outcomes, chat.chat_template_tokens if chat else 0)
    summary = {
        "evidence_type": arguments.evidence_type,
        "started_at_wall": started_at_wall,
        "observed_seconds": time.monotonic() - start,
        "what_this_is": (
            "recorded agent sessions (token counts and tool waits) replayed against a "
            "real engine with synthetic token ids; tool waits are sleeps"
        ),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(arguments).items()
        },
        "revision": json.loads(revision_path.read_text())
        if revision_path.exists()
        else None,
        "code_provenance": git_state(),
        "replayer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sessions_sha256": hashlib.sha256(
            (arguments.out / "sessions.csv").read_bytes()
        ).hexdigest(),
        "engines": engines,
        "chat_template_tokens": chat.chat_template_tokens if chat else 0,
        "prompt_token_counting": "Exact full per-request chat template; empty-template size is reference only",
        "result": result,
        "workload_manifest": json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else None,
    }
    (arguments.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(result, indent=1), flush=True)
    print(f"[sessions] wrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
