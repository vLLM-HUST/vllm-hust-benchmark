"""Shared canonical rules for strict host execution evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


CANONICAL_WORKER_RULE = (
    "per-physical-npu: prefer cmdline matching VLLMWorker/Worker_TP, "
    "then EngineCore, then other owned compute processes; tie-break by lowest host PID"
)
STRICT_V018_RUNTIME_PYTHON = "/usr/local/python3.11.14/bin/python"
OWNED_RUNTIME_SECURITY_SCHEMA = "owned-runtime-security/v1"
OWNED_RUNTIME_AUTHORIZATION_SOURCE_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,511}"
OWNED_RUNTIME_PREFLIGHT = (
    "/workspace/vllm-hust-benchmark/scripts/verify-owned-runtime-and-exec.py"
)
STRICT_ASCEND_READONLY_MOUNTS = (
    ("/usr/local/Ascend/driver", "directory"),
    ("/etc/ascend_install.info", "file"),
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
