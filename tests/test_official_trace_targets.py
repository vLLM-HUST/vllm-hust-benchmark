from __future__ import annotations

import csv
import gzip
import json
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema
import pytest

from vllm_hust_benchmark.official_trace_targets import (
    deterministic_token_ids,
    iter_trace_requests,
    load_trace_target_registry,
)
from vllm_hust_benchmark.trace_replay import (
    execute_replay,
    load_replay_plan,
    summarize_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BURSTGPT = "burstgpt-v2-production-replay"
TRACELAB = "tracelab-v0.0.1-coding-agent-replay"


def _write_burstgpt(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Timestamp",
                "Session ID",
                "Elapsed time",
                "Model",
                "Request tokens",
                "Response tokens",
                "Total tokens",
                "Log Type",
            ]
        )
        writer.writerow([10, "session-a", 3.5, "GPT-4", 100, 20, 120, "API log"])
        writer.writerow([12, "session-a", 4.0, "GPT-4", 140, 0, 140, "API log"])
        writer.writerow([11, "session-b", 2.0, "ChatGPT", 80, 10, 90, "API log"])


def _write_tracelab(path: Path) -> None:
    rows = [
        {
            "provider": "codex",
            "session_id": "session-c",
            "trace_key": "trace-1",
            "model": "gpt-5",
            "input_tokens_total": 120,
            "prefix_tokens": 100,
            "newly_append_tokens": 20,
            "output_tokens": 12,
            "current_user_message_count": 0,
            "current_tool_result_count": 1,
            "timing_events": [
                {"event_type": "tool_result", "timestamp": "2026-01-01T00:00:05Z"}
            ],
        },
        {
            "provider": "claude",
            "session_id": "session-d",
            "trace_key": "trace-2",
            "model": "claude-opus",
            "input_tokens_total": 150,
            "prefix_tokens": 120,
            "newly_append_tokens": 30,
            "output_tokens": 15,
            "current_user_message_count": 1,
            "current_tool_result_count": 0,
            "timing_events": [
                {"event_type": "user_message", "timestamp": "2026-01-01T00:00:07Z"}
            ],
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_trace_target_registry_matches_schema() -> None:
    registry = load_trace_target_registry()
    schema = json.loads(
        (
            REPO_ROOT / "schemas" / "official_trace_target_registry_v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(registry)


def test_trace_targets_are_pinned_and_provisional() -> None:
    targets = load_trace_target_registry()["targets"]
    assert {target["id"] for target in targets} == {BURSTGPT, TRACELAB}
    assert {target["status"] for target in targets} == {"provisional"}
    for target in targets:
        assert target["source"]["release"] != "latest"
        assert len(target["source"]["sha256"]) == 64
        assert target["source"]["license"] == "CC-BY-4.0"
        assert (
            target["replay_contract"]["payload"] == "deterministic-synthetic-token-ids"
        )


def test_burstgpt_parser_preserves_arrival_session_tokens_and_latency(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "BurstGPT_3.csv"
    _write_burstgpt(trace)

    requests = list(iter_trace_requests(BURSTGPT, trace))

    assert requests[0].session_id == "session-a"
    assert requests[0].arrival_s == 10
    assert requests[0].input_tokens == 100
    assert requests[0].output_tokens == 20
    assert requests[0].observed_latency_s == 3.5
    assert requests[1].output_tokens == 0


def test_tracelab_parser_preserves_prefix_append_and_trigger(tmp_path: Path) -> None:
    trace = tmp_path / "syfi_coding_trace.jsonl.gz"
    _write_tracelab(trace)

    requests = list(iter_trace_requests(TRACELAB, trace))

    assert requests[0].prefix_tokens == 100
    assert requests[0].append_tokens == 20
    assert requests[0].input_tokens == 120
    assert requests[0].trigger == "tool_result"
    assert requests[1].trigger == "user_message"


def test_plan_sorts_arrivals_scales_time_and_counts_source_failures(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "BurstGPT_3.csv"
    _write_burstgpt(trace)

    plan, source_failures = load_replay_plan(
        BURSTGPT,
        trace,
        max_requests=2,
        max_model_len=256,
        overflow_policy="reject",
        time_scale=2,
        max_interarrival_s=None,
    )
    summary = summarize_plan(BURSTGPT, plan, source_failures=source_failures)

    assert [item.request.request_id for item in plan] == ["burstgpt-0", "burstgpt-2"]
    assert [item.scheduled_offset_s for item in plan] == [0, 0.5]
    assert source_failures == 1
    assert summary["requests"] == 2
    assert summary["sessions"] == 2


def test_context_overflow_fails_closed_and_requires_explicit_truncation(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "syfi_coding_trace.jsonl.gz"
    _write_tracelab(trace)

    with pytest.raises(ValueError, match="exceeding max_model_len"):
        load_replay_plan(
            TRACELAB,
            trace,
            max_requests=1,
            max_model_len=125,
            overflow_policy="reject",
            time_scale=1,
            max_interarrival_s=None,
        )

    plan, _ = load_replay_plan(
        TRACELAB,
        trace,
        max_requests=1,
        max_model_len=125,
        overflow_policy="truncate-input",
        time_scale=1,
        max_interarrival_s=None,
    )
    assert plan[0].replay_input_tokens == 113


def test_synthetic_payload_is_session_stable_and_cross_session_distinct(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "BurstGPT_3.csv"
    _write_burstgpt(trace)
    requests = list(iter_trace_requests(BURSTGPT, trace))

    first = deterministic_token_ids(requests[0])
    same_session = deterministic_token_ids(requests[1])
    other_session = deterministic_token_ids(requests[2])

    assert same_session[: len(first)] == first
    assert other_session[:10] != first[:10]


def test_tracelab_synthetic_prefix_is_stable_and_append_is_fresh(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "syfi_coding_trace.jsonl.gz"
    _write_tracelab(trace)
    request = list(iter_trace_requests(TRACELAB, trace))[0]
    next_request = replace(request, request_id="trace-next")

    first = deterministic_token_ids(request)
    second = deterministic_token_ids(next_request)

    assert first[:100] == second[:100]
    assert first[100:] != second[100:]


def test_execute_replay_sends_exact_synthetic_token_count(tmp_path: Path) -> None:
    trace = tmp_path / "BurstGPT_3.csv"
    _write_burstgpt(trace)
    plan, _ = load_replay_plan(
        BURSTGPT,
        trace,
        max_requests=1,
        max_model_len=256,
        overflow_policy="reject",
        time_scale=1,
        max_interarrival_s=None,
    )
    captured: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            captured.append(json.loads(self.rfile.read(length)))
            body = json.dumps(
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "total_tokens": 120,
                    }
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        results = execute_replay(
            plan,
            base_url=f"http://127.0.0.1:{server.server_port}",
            endpoint="/v1/completions",
            model="test-model",
            api_key="test-key",
            max_concurrency=1,
            timeout_s=5,
            token_id_min=1000,
            token_id_max=30000,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert results[0]["http_status"] == 200
    assert results[0]["response_usage"]["total_tokens"] == 120
    assert len(captured[0]["prompt"]) == 100
    assert captured[0]["max_tokens"] == 20
    assert captured[0]["model"] == "test-model"
