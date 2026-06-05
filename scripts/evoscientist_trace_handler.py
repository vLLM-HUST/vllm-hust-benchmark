#!/usr/bin/env python3
"""LangChain callback handler for capturing EvoScientist workload traces.

This module provides a WorkloadTraceHandler that records every LLM call
(prompt, completion, token counts, latency) to a JSONL file. The resulting
trace can then be converted into a vllm-hust-benchmark workload dataset.

Usage:
    from evoscientist_trace_handler import WorkloadTraceHandler

    handler = WorkloadTraceHandler("traces/my-trace.jsonl")
    # Pass as callback to LangGraph config:
    config = {"configurable": {"thread_id": "1"}, "callbacks": [handler]}
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult


def _serialize_messages(messages: list[list[BaseMessage]]) -> list[list[dict]]:
    """Convert LangChain message objects to serializable dicts."""
    result = []
    for batch in messages:
        batch_serialized = []
        for msg in batch:
            entry: dict[str, Any] = {
                "role": msg.type,  # "human", "ai", "system", "tool"
                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
            }
            if hasattr(msg, "name") and msg.name:
                entry["name"] = msg.name
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = [
                    {"name": tc["name"], "args": tc["args"]}
                    for tc in msg.tool_calls
                ]
            batch_serialized.append(entry)
        result.append(batch_serialized)
    return result


def _extract_completion_text(result: LLMResult) -> str:
    """Extract the completion text from an LLMResult."""
    if not result.generations:
        return ""
    first_gen = result.generations[0]
    if not first_gen:
        return ""
    gen = first_gen[0]
    if hasattr(gen, "message") and gen.message:
        content = gen.message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)
    return gen.text if hasattr(gen, "text") else str(gen)


def _extract_token_usage(result: LLMResult) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from LLMResult metadata."""
    input_tokens = 0
    output_tokens = 0

    # Try llm_output first
    if result.llm_output:
        usage = result.llm_output.get("token_usage") or result.llm_output.get("usage", {})
        if usage:
            input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

    # Fallback: check generation metadata
    if input_tokens == 0 and output_tokens == 0 and result.generations:
        for gen_batch in result.generations:
            for gen in gen_batch:
                info = getattr(gen, "generation_info", None) or {}
                usage = info.get("usage", {})
                if usage:
                    input_tokens += usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                    output_tokens += usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)

    return input_tokens, output_tokens


class WorkloadTraceHandler(BaseCallbackHandler):
    """Captures all LLM calls to a JSONL trace file for workload analysis.

    Each line in the output file is a JSON object with:
    - timestamp: ISO-8601 UTC timestamp
    - run_id: LangChain run UUID (links sub-agent calls)
    - parent_run_id: Parent run UUID (identifies delegation chain)
    - prompt_messages: The input messages sent to the LLM
    - completion: The LLM response text
    - input_tokens: Number of prompt tokens
    - output_tokens: Number of completion tokens
    - latency_ms: Wall-clock latency in milliseconds
    - model: Model name if available
    - metadata: Any additional metadata from the run
    """

    name = "WorkloadTraceHandler"

    def __init__(self, output_path: str | Path, flush_every: int = 1) -> None:
        """Initialize the trace handler.

        Args:
            output_path: Path to the JSONL output file.
            flush_every: Flush to disk every N entries (default: every entry).
        """
        super().__init__()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._lock = threading.Lock()
        self._pending_starts: dict[str, dict[str, Any]] = {}
        self._entries_since_flush = 0
        self._file = open(self.output_path, "a", encoding="utf-8")
        self._total_entries = 0

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of an LLM call (text-based)."""
        with self._lock:
            self._pending_starts[str(run_id)] = {
                "start_time": time.perf_counter(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "prompts": prompts,
                "messages": None,
                "tags": tags or [],
                "metadata": metadata or {},
                "model": serialized.get("kwargs", {}).get("model_name", ""),
            }

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the start of a chat model call (message-based)."""
        with self._lock:
            self._pending_starts[str(run_id)] = {
                "start_time": time.perf_counter(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "parent_run_id": str(parent_run_id) if parent_run_id else None,
                "prompts": None,
                "messages": _serialize_messages(messages),
                "tags": tags or [],
                "metadata": metadata or {},
                "model": (
                    serialized.get("kwargs", {}).get("model_name", "")
                    or serialized.get("kwargs", {}).get("model", "")
                ),
            }

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record the end of an LLM call and write the trace entry."""
        with self._lock:
            run_key = str(run_id)
            start_info = self._pending_starts.pop(run_key, None)
            if start_info is None:
                return

            elapsed_ms = (time.perf_counter() - start_info["start_time"]) * 1000
            completion = _extract_completion_text(response)
            input_tokens, output_tokens = _extract_token_usage(response)

            entry = {
                "timestamp": start_info["timestamp"],
                "run_id": run_key,
                "parent_run_id": start_info["parent_run_id"],
                "model": start_info["model"],
                "prompt_messages": start_info["messages"] or start_info["prompts"],
                "completion": completion,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": round(elapsed_ms, 2),
                "tags": start_info["tags"],
                "metadata": start_info["metadata"],
            }

            self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._entries_since_flush += 1
            self._total_entries += 1

            if self._entries_since_flush >= self.flush_every:
                self._file.flush()
                self._entries_since_flush = 0

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Clean up pending start on error."""
        with self._lock:
            self._pending_starts.pop(str(run_id), None)

    @property
    def total_entries(self) -> int:
        """Number of trace entries written so far."""
        return self._total_entries

    def close(self) -> None:
        """Flush and close the output file."""
        with self._lock:
            self._file.flush()
            self._file.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
