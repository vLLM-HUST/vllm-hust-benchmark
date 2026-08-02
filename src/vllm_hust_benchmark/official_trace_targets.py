from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = "official-trace-target-registry/v1"


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    session_id: str
    arrival_s: float
    input_tokens: int
    output_tokens: int
    prefix_tokens: int | None
    append_tokens: int | None
    observed_latency_s: float | None
    source_model: str | None
    provider: str | None
    trigger: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_trace_target_registry() -> dict[str, Any]:
    with (
        resources.files("vllm_hust_benchmark.data")
        .joinpath("official_trace_targets.json")
        .open("r", encoding="utf-8") as handle
    ):
        payload = json.load(handle)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported official trace target registry schema")
    targets = payload.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("official trace target registry is empty")
    ids = [target.get("id") for target in targets]
    if len(ids) != len(set(ids)):
        raise ValueError("official trace target IDs must be unique")
    for target in targets:
        source = target.get("source") or {}
        digest = source.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid source checksum for {target.get('id')!r}")
        if target.get("status") == "active":
            raise ValueError(
                "trace targets require real matched baseline evidence before activation"
            )
    return payload


def get_trace_target(target_id: str) -> dict[str, Any]:
    for target in load_trace_target_registry()["targets"]:
        if target["id"] == target_id:
            return target
    raise KeyError(f"unknown official trace target: {target_id}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_trace_asset(target: dict[str, Any], path: Path) -> None:
    source = target["source"]
    if path.stat().st_size != int(source["size_bytes"]):
        raise ValueError(
            f"trace asset size mismatch for {target['id']}: "
            f"expected {source['size_bytes']}, got {path.stat().st_size}"
        )
    actual = sha256_file(path)
    if actual != source["sha256"]:
        raise ValueError(
            f"trace asset checksum mismatch for {target['id']}: "
            f"expected {source['sha256']}, got {actual}"
        )


def fetch_trace_asset(target_id: str, cache_dir: Path) -> Path:
    target = get_trace_target(target_id)
    source = target["source"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / source["asset"]
    if destination.exists():
        verify_trace_asset(target, destination)
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with (
            urllib.request.urlopen(source["url"]) as response,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output)
        verify_trace_asset(target, partial)
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _as_int(value: Any, field: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if result < 0:
        raise ValueError(f"negative {field}: {result}")
    return result


def _as_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field}: {value!r}") from exc
    if result < 0:
        raise ValueError(f"negative {field}: {result}")
    return result


def _burstgpt_requests(path: Path) -> Iterator[TraceRequest]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            session = str(row.get("Session ID") or f"request-{index}")
            latency = row.get("Elapsed time")
            yield TraceRequest(
                request_id=f"burstgpt-{index}",
                session_id=session,
                arrival_s=_as_float(row.get("Timestamp"), "Timestamp"),
                input_tokens=_as_int(row.get("Request tokens"), "Request tokens"),
                output_tokens=_as_int(row.get("Response tokens"), "Response tokens"),
                prefix_tokens=None,
                append_tokens=None,
                observed_latency_s=(
                    _as_float(latency, "Elapsed time")
                    if latency not in (None, "")
                    else None
                ),
                source_model=str(row.get("Model") or "") or None,
                provider="azure-openai",
                trigger=None,
            )


def _event_arrival_s(row: dict[str, Any]) -> float:
    timestamps = [
        event.get("timestamp")
        for event in row.get("timing_events") or []
        if isinstance(event, dict) and event.get("timestamp")
    ]
    if not timestamps:
        raise ValueError(f"TraceLab row has no timing event: {row.get('trace_key')}")
    value = min(str(timestamp) for timestamp in timestamps)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _tracelab_trigger(row: dict[str, Any]) -> str:
    if _as_int(row.get("current_tool_result_count", 0), "current_tool_result_count"):
        return "tool_result"
    if _as_int(row.get("current_user_message_count", 0), "current_user_message_count"):
        return "user_message"
    return str(row.get("first_input_event_type") or "other")


def _tracelab_requests(path: Path) -> Iterator[TraceRequest]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            prefix = _as_int(row.get("prefix_tokens"), "prefix_tokens")
            append = _as_int(row.get("newly_append_tokens"), "newly_append_tokens")
            input_tokens = _as_int(row.get("input_tokens_total"), "input_tokens_total")
            if prefix + append != input_tokens:
                raise ValueError(
                    f"TraceLab token accounting mismatch at row {index}: "
                    f"{prefix} + {append} != {input_tokens}"
                )
            yield TraceRequest(
                request_id=str(row.get("trace_key") or f"tracelab-{index}"),
                session_id=str(row.get("session_id") or f"session-{index}"),
                arrival_s=_event_arrival_s(row),
                input_tokens=input_tokens,
                output_tokens=_as_int(row.get("output_tokens"), "output_tokens"),
                prefix_tokens=prefix,
                append_tokens=append,
                observed_latency_s=None,
                source_model=str(row.get("model") or "") or None,
                provider=str(row.get("provider") or "") or None,
                trigger=_tracelab_trigger(row),
            )


def iter_trace_requests(target_id: str, path: Path) -> Iterator[TraceRequest]:
    get_trace_target(target_id)
    if target_id == "burstgpt-v2-production-replay":
        yield from _burstgpt_requests(path)
        return
    if target_id == "tracelab-v0.0.1-coding-agent-replay":
        yield from _tracelab_requests(path)
        return
    raise KeyError(f"no trace parser for target: {target_id}")


def deterministic_token_ids(
    request: TraceRequest,
    *,
    input_tokens: int | None = None,
    token_id_min: int = 1000,
    token_id_max: int = 30000,
) -> list[int]:
    if token_id_min < 0 or token_id_max <= token_id_min:
        raise ValueError("invalid synthetic token ID range")
    length = request.input_tokens if input_tokens is None else input_tokens
    session_seed = int.from_bytes(
        hashlib.sha256(request.session_id.encode("utf-8")).digest()[:8], "big"
    )
    request_seed = int.from_bytes(
        hashlib.sha256(request.request_id.encode("utf-8")).digest()[:8], "big"
    )
    prefix_tokens = request.prefix_tokens
    if prefix_tokens is None:
        prefix_tokens = length
    prefix_tokens = min(prefix_tokens, length)
    width = token_id_max - token_id_min + 1
    return [
        token_id_min
        + (
            (session_seed if position < prefix_tokens else request_seed)
            + position * 104729
        )
        % width
        for position in range(length)
    ]
