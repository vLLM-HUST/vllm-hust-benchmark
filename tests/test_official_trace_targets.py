from __future__ import annotations

import csv
import gzip
import json
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jsonschema
import pytest

from vllm_hust_benchmark.official_trace_targets import (
    TraceRequest,
    iter_trace_requests,
    load_trace_target_registry,
)
from vllm_hust_benchmark.trace_replay import (
    PlannedRequest,
    build_prompt_token_ids,
    execute_replay,
    load_replay_plan,
    summarize_plan,
    summarize_results,
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


def test_trace_targets_are_pinned_and_active_in_their_dedicated_profile() -> None:
    targets = load_trace_target_registry()["targets"]
    assert {target["id"] for target in targets} == {BURSTGPT, TRACELAB}
    assert {target["status"] for target in targets} == {"active"}
    for target in targets:
        assert target["source"]["release"] != "latest"
        assert len(target["source"]["sha256"]) == 64
        assert target["source"]["license"] == "CC-BY-4.0"
        assert target["intended_use"] == "public-leaderboard-production-trace"
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
        max_requests=3,
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


def test_plan_summary_has_token_distributions_and_stable_sensitive_signature(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "BurstGPT_3.csv"
    _write_burstgpt(trace)
    plan, source_failures = load_replay_plan(
        BURSTGPT,
        trace,
        max_requests=3,
        max_model_len=256,
        overflow_policy="reject",
        time_scale=2,
        max_interarrival_s=None,
    )
    settings = {
        "trace_asset_sha256": "a" * 64,
        "max_requests": 3,
        "max_model_len": 256,
        "overflow_policy": "reject",
        "time_scale": 2,
        "max_interarrival_s": None,
        "provider": None,
    }

    summary = summarize_plan(
        BURSTGPT, plan, source_failures=source_failures, **settings
    )
    repeated = summarize_plan(
        BURSTGPT, plan, source_failures=source_failures, **settings
    )
    reordered = summarize_plan(
        BURSTGPT,
        list(reversed(plan)),
        source_failures=source_failures,
        **settings,
    )
    changed_settings = summarize_plan(
        BURSTGPT,
        plan,
        source_failures=source_failures,
        **{**settings, "time_scale": 3},
    )

    assert summary["input_tokens"] == {
        "count": 2,
        "min": 80,
        "p50": 90.0,
        "p95": 99.0,
        "p99": 99.8,
        "max": 100,
        "total": 180,
    }
    assert summary["output_tokens"] == {
        "count": 2,
        "min": 10,
        "p50": 15.0,
        "p95": 19.5,
        "p99": 19.9,
        "max": 20,
        "total": 30,
    }
    assert len(summary["selected_requests_sha256"]) == 64
    assert len(summary["cohort_setting_signature"]) == 64
    assert repeated["cohort_setting_signature"] == summary[
        "cohort_setting_signature"
    ]
    assert reordered["selected_requests_sha256"] != summary[
        "selected_requests_sha256"
    ]
    assert reordered["cohort_setting_signature"] != summary[
        "cohort_setting_signature"
    ]
    assert changed_settings["selected_requests_sha256"] == summary[
        "selected_requests_sha256"
    ]
    assert changed_settings["cohort_setting_signature"] != summary[
        "cohort_setting_signature"
    ]


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


def test_context_window_cohort_excludes_and_counts_overflow(tmp_path: Path) -> None:
    trace = tmp_path / "syfi_coding_trace.jsonl.gz"
    _write_tracelab(trace)
    stats: dict[str, int] = {}

    plan, failures = load_replay_plan(
        TRACELAB,
        trace,
        max_requests=2,
        max_model_len=140,
        overflow_policy="exclude-overflow",
        time_scale=1,
        max_interarrival_s=None,
        exclusion_stats=stats,
    )

    assert failures == 0
    assert [item.request.request_id for item in plan] == ["trace-1"]
    assert stats["context_overflow_rows_excluded"] == 1


def test_burstgpt_payload_is_request_specific_even_within_one_session(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "BurstGPT_3.csv"
    _write_burstgpt(trace)
    with trace.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            [13, "session-a", 1.0, "GPT-4", 120, 5, 125, "API log"]
        )
    plan, _ = load_replay_plan(
        BURSTGPT,
        trace,
        max_requests=3,
        max_model_len=256,
        overflow_policy="reject",
        time_scale=1,
        max_interarrival_s=None,
    )
    same_session = [item for item in plan if item.request.session_id == "session-a"]

    first = build_prompt_token_ids(same_session[0])
    second = build_prompt_token_ids(same_session[1])

    assert first[0] != second[0]
    assert same_session[1].effective_prefix_tokens == 0


def test_tracelab_continuous_session_replays_reported_prefix_as_actual_lcp(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "syfi_coding_trace.jsonl.gz"
    # Exact prefix/append counts from two consecutive rows in the public trace.
    rows = [
        {
            "provider": "claude",
            "session_id": "claude:f4b60bc0",
            "trace_key": "round-e14eb886",
            "model": "claude-opus",
            "input_tokens_total": 14438,
            "prefix_tokens": 0,
            "newly_append_tokens": 14438,
            "output_tokens": 157,
            "current_user_message_count": 1,
            "current_tool_result_count": 0,
            "timing_events": [
                {"event_type": "user_message", "timestamp": "2026-04-18T08:23:07.368Z"}
            ],
        },
        {
            "provider": "claude",
            "session_id": "claude:f4b60bc0",
            "trace_key": "round-b5c70f1",
            "model": "claude-opus",
            "input_tokens_total": 16159,
            "prefix_tokens": 14435,
            "newly_append_tokens": 1724,
            "output_tokens": 91,
            "current_user_message_count": 0,
            "current_tool_result_count": 1,
            "timing_events": [
                {"event_type": "tool_result", "timestamp": "2026-04-18T08:23:12.851Z"}
            ],
        },
    ]
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    plan, _ = load_replay_plan(
        TRACELAB,
        trace,
        max_requests=2,
        max_model_len=32768,
        overflow_policy="reject",
        time_scale=1,
        max_interarrival_s=None,
    )

    first = build_prompt_token_ids(plan[0])
    second = build_prompt_token_ids(plan[1])
    actual_lcp = next(
        (index for index, pair in enumerate(zip(first, second)) if pair[0] != pair[1]),
        min(len(first), len(second)),
    )

    assert plan[1].effective_prefix_tokens == 14435
    assert actual_lcp == 14435


def test_tracelab_cohort_is_selected_from_global_arrival_order(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "syfi_coding_trace.jsonl.gz"
    rows = []
    for request_id, second in (("late", 9), ("middle", 5), ("early", 1)):
        rows.append(
            {
                "provider": "codex",
                "session_id": request_id,
                "trace_key": request_id,
                "model": "gpt-5",
                "input_tokens_total": 10,
                "prefix_tokens": 0,
                "newly_append_tokens": 10,
                "output_tokens": 1,
                "current_user_message_count": 1,
                "current_tool_result_count": 0,
                "timing_events": [
                    {
                        "event_type": "user_message",
                        "timestamp": f"2026-01-01T00:00:0{second}Z",
                    }
                ],
            }
        )
    with gzip.open(trace, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    plan, _ = load_replay_plan(
        TRACELAB,
        trace,
        max_requests=2,
        max_model_len=128,
        overflow_policy="reject",
        time_scale=1,
        max_interarrival_s=None,
    )

    assert [item.request.request_id for item in plan] == ["early", "middle"]


def test_summary_includes_executor_queue_delay_and_actual_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = TraceRequest(
        request_id="queued",
        session_id="queue-session",
        arrival_s=0,
        input_tokens=1,
        output_tokens=3,
        prefix_tokens=None,
        append_tokens=None,
        observed_latency_s=None,
        source_model=None,
        provider="azure-openai",
        trigger=None,
    )
    plan = [
        PlannedRequest(
            request=replace(request, request_id=f"queued-{index}"),
            scheduled_offset_s=0,
            replay_input_tokens=1,
        )
        for index in range(3)
    ]

    def fake_invoke(
        item: PlannedRequest, *, replay_started: float, **_kwargs: object
    ) -> dict[str, object]:
        actual_started = time.monotonic()
        time.sleep(0.04)
        finished = time.monotonic()
        return {
            "http_status": 200,
            "scheduled_offset_s": item.scheduled_offset_s,
            "actual_start_offset_s": actual_started - replay_started,
            "finished_offset_s": finished - replay_started,
            "queue_delay_s": actual_started - replay_started,
            "replay_latency_s": finished - actual_started,
            "e2e_latency_s": finished - replay_started,
            "response_usage": {"prompt_tokens": 2, "completion_tokens": 3},
        }

    monkeypatch.setattr("vllm_hust_benchmark.trace_replay._invoke", fake_invoke)
    results = execute_replay(
        plan,
        base_url="http://127.0.0.1:1",
        endpoint="/v1/completions",
        model="test-model",
        api_key=None,
        max_concurrency=1,
        timeout_s=1,
        token_id_min=1000,
        token_id_max=30000,
    )
    summary = summarize_results(plan, results)

    assert summary["duration"] >= 0.1
    assert summary["request_throughput"] < 30
    assert summary["total_input_tokens"] == 6
    assert summary["total_output_tokens"] == 9
    assert results[-1]["queue_delay_s"] >= 0.07
    assert summary["p95_e2e_latency_ms"] >= summary["p50_e2e_latency_ms"]


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
            api_key="test-key",  # pragma: allowlist secret
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
    summary = summarize_results(plan, results)
    assert summary["completed"] == 1
    assert summary["failed"] == 0
    assert summary["total_output_tokens"] == 20
    assert summary["output_throughput"] > 0
    assert "mean_ttft_ms" not in summary
    assert len(captured[0]["prompt"]) == 100
    assert captured[0]["max_tokens"] == 20
    assert captured[0]["model"] == "test-model"


def test_execute_replay_rejects_inexact_response_usage(tmp_path: Path) -> None:
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

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers["Content-Length"])
            json.loads(self.rfile.read(length))
            body = json.dumps(
                {
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 19,
                        "total_tokens": 119,
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
            api_key=None,
            max_concurrency=1,
            timeout_s=5,
            token_id_min=1000,
            token_id_max=30000,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert "completion_tokens does not match" in results[0]["error"]
    summary = summarize_results(plan, results)
    assert summary["completed"] == 0
    assert summary["failed"] == 1
