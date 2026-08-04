"""Shared canonical rules for strict host execution evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


CANONICAL_WORKER_RULE = (
    "per-physical-npu: prefer cmdline matching VLLMWorker/Worker_TP, "
    "then EngineCore, then other owned compute processes; tie-break by lowest host PID"
)
_WORKER_RE = re.compile(r"(?:vllmworker|worker_tp)", re.IGNORECASE)
_ENGINE_CORE_RE = re.compile(r"enginecore", re.IGNORECASE)


def canonical_worker_key(record: Mapping[str, Any]) -> tuple[int, int]:
    """Return the deterministic canonical ordering for one owned compute PID."""
    cmdline = str(record.get("cmdline") or "")
    if _WORKER_RE.search(cmdline):
        role_rank = 0
    elif _ENGINE_CORE_RE.search(cmdline):
        role_rank = 1
    else:
        role_rank = 2
    host_pid = record.get("host_pid")
    if not isinstance(host_pid, int) or host_pid <= 0:
        raise ValueError("owned compute process has an invalid host PID")
    return role_rank, host_pid
