"""Watchdog ops closed-loop for the poy-180 NPU runner watchdog (issue #125).

The watchdog daemon itself scans NPU 0-4 every 30 seconds and reclaims
non-runner processes, but it lives on a private runner host and is not part of
this repository. This module owns the *auditability* half of the closed-loop so
that the public repo can enforce and verify the contract:

- **JSONL event schema**: every scan decision is one JSONL audit record
  conforming to ``schemas/npu_watchdog_event_v1.schema.json``.
- **Determination + result fields**: each record carries the ownership
  ``determination`` (``runner-job`` / ``sibling-container`` /
  ``unauthorized-container`` / ``unowned-process``) and the exit/cleanup
  ``result`` (``no-op`` / ``exited-before-action`` / ``terminated`` /
  ``killed`` / ``not-found``), which is what the issue's operator comment asks
  to make explicit.
- **Alert dedup + owner + recovery status**: ``dedup_key`` stably identifies
  one event; ``should_alert`` suppresses the same unchanged event while it
  stays open and re-alerts on state transitions; ``owner`` and
  ``recovery_status`` are carried on every record.
- **Host JSONL <-> GitHub summary consistency**: the GitHub comment is rendered
  *from* the JSONL record (single source of truth) and
  ``verify_summary_consistency`` re-parses a rendered summary to prove they
  match.
- **NPU 4 policy**: NPU 4 has no registered runner, so any occupancy is a
  violation and is flagged with ``npu4_unregistered_runner``.

Command lines are never recorded: only ``cmdline_sha256`` is kept so the same
process can be recognised across scans without leaking arguments that may
contain secrets.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, FormatChecker

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "npu_watchdog_event_v1.schema.json"
)
SCHEMA_VERSION = "npu-watchdog-event/v1"
SCHEMA_NAME = "npu-watchdog-event"

# poy-180 monitoring scope.
NPU_MIN = 0
NPU_MAX = 4
# Registered poy-180 runner physical devices. NPU 4 has no runner.
REGISTERED_RUNNER_NPUS = frozenset({0, 1, 2, 3})

DETERMINATIONS = frozenset(
    {"runner-job", "sibling-container", "unauthorized-container", "unowned-process"}
)
ACTIONS = frozenset({"none", "sigterm", "sigkill"})
RESULTS = frozenset(
    {"no-op", "exited-before-action", "terminated", "killed", "not-found"}
)
RECOVERY_STATUSES = frozenset({"open", "recovered"})

# Defaults for the ops closed-loop; the daemon may override owner / mention.
DEFAULT_OWNER = "SuccinctPaul"
DEFAULT_ALERT_MENTION = "@ShuhaoZhangTony"
DEFAULT_EVENTS_PATH = "/var/log/npu-runner-watchdog/events.jsonl"
DEFAULT_ISSUE = 125
DEFAULT_REPO = "vLLM-HUST/vllm-hust-benchmark"

# Runner containers and their sibling containers carry org.vllm-hust.runner
# labels like ``poy-180-21rc-npu0`` … ``poy-180-21rc-npu3``.
RUNNER_LABEL = "org.vllm-hust.runner"
RUNNER_NAME_PATTERN = re.compile(r"^poy-180-21rc-npu(?P<device>[0-3])$")

_VALIDATOR: Draft7Validator | None = None


def _validator() -> Draft7Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = Draft7Validator(
            json.loads(SCHEMA_PATH.read_text(encoding="utf-8")),
            format_checker=FormatChecker(),
        )
    return _VALIDATOR


def derive_cmdline_sha256(cmdline: Iterable[str] | None) -> str:
    """Hash a command line (or None) into the recorded ``cmdline_sha256``.

    The raw command line is never stored; only this digest is kept, so the
    same process can be recognised across scans without leaking arguments.
    """
    if cmdline is None:
        payload = b""
    else:
        payload = "\x00".join(cmdline).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def derive_dedup_key(npu: int, pid: int, cmdline_sha256: str) -> str:
    """Stable identity for one watchdog event across scans.

    A process is "the same event" while it keeps the same ``(npu, pid)`` and a
    matching command-line digest. If a PID is reused by a different process the
    digest differs and a new event is started.
    """
    return f"npu{npu}/pid{pid}/cmd{cmdline_sha256[:12]}"


def derive_recovery_status(result: str) -> str:
    """Return the recovery status implied by an exit/cleanup result.

    A violation is ``recovered`` once the offending process is gone: we
    terminated/killed it, it exited before our action, or it was already gone
    (``not-found``). A ``no-op`` record (owned process left alone) stays
    ``open`` because there is no violation to close out.
    """
    if result in ("terminated", "killed", "not-found", "exited-before-action"):
        return "recovered"
    return "open"


def validate_event_record(record: Mapping[str, Any]) -> list[str]:
    """Validate one event record against the v1 schema; return error strings.

    Returns an empty list when the record conforms.
    """
    errors: list[str] = []
    for error in sorted(
        _validator().iter_errors(dict(record)), key=lambda item: tuple(item.path)
    ):
        path = ".".join(str(part) for part in error.path) or "<root>"
        errors.append(f"{path}: {error.message}")
    return errors


def npu_is_policy_violation(npu: int) -> bool:
    """Return True when an NPU has no registered runner and is thus a violation.

    NPU 4 currently has no registered runner, so any occupancy is treated as a
    violation (explicit policy requested by issue #125).
    """
    return npu not in REGISTERED_RUNNER_NPUS


def classify_determination(
    *,
    npu: int,
    container_name: str | None,
    container_runner_label: str | None,
    npu_physical_label: str | None,
) -> str:
    """Classify the ownership determination of one NPU-occupying process.

    ``container_name`` is ``None`` for host (non-container) processes.

    - ``runner-job``: the process runs inside the registered runner container
      itself (``poy-180-21rc-npuN``) for the NPU it occupies.
    - ``sibling-container``: a sibling container carrying the matching
      ``org.vllm-hust.runner`` label and the same physical NPU mapping.
    - ``unauthorized-container``: inside a container but the ownership label /
      device mapping does not match the NPU it occupies.
    - ``unowned-process``: not inside any runner/sibling container.
    """
    if container_name is None:
        return "unowned-process"
    runner_match = RUNNER_NAME_PATTERN.match(container_runner_label or "")
    if runner_match is None:
        return "unauthorized-container"
    if int(runner_match.group("device")) != npu:
        return "unauthorized-container"
    if npu_physical_label is not None and str(npu_physical_label) != str(npu):
        return "unauthorized-container"
    if container_name == container_runner_label:
        return "runner-job"
    return "sibling-container"


def should_alert(
    record: Mapping[str, Any],
    previous_alerts: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, str]:
    """Decide whether an event record should trigger a new GitHub alert.

    ``previous_alerts`` maps ``dedup_key`` to the last *alerted* record for
    that event (only ``result`` and ``recovery_status`` are consulted).

    Returns ``(should_alert, reason)``. The same event is suppressed while it
    stays open and unchanged; it re-alerts when its recovery status changes
    (open -> recovered closure), when the result escalates (e.g.
    ``terminated`` -> ``killed``), or when a fresh occurrence appears.
    """
    key = str(record.get("dedup_key") or "")
    if not key:
        return True, "event without dedup_key"
    previous = previous_alerts.get(key)
    if previous is None:
        return True, "new event"
    prev_result = previous.get("result")
    prev_recovery = previous.get("recovery_status")
    result = record.get("result")
    recovery = record.get("recovery_status")
    if prev_recovery != recovery:
        return True, f"recovery status changed {prev_recovery!r} -> {recovery!r}"
    if prev_result != result:
        return True, f"result changed {prev_result!r} -> {result!r}"
    return False, "unchanged open event already alerted"


def _display(value: Any) -> str:
    """Render a nullable field as 'unknown' instead of None in the summary."""
    if value is None:
        return "unknown"
    return str(value)


def render_github_summary(
    record: Mapping[str, Any], event_line: int | None = None
) -> str:
    """Render the GitHub issue comment for one event record.

    The comment is derived purely from the JSONL record, so the host JSONL
    audit and the posted summary cannot diverge. It never includes the raw
    command line.
    """
    lines: list[str] = []
    lines.append(
        f"{DEFAULT_ALERT_MENTION}，NPU runner watchdog 在 `{record['host']}` "
        "处置了非 Runner 进程。"
    )
    lines.append("")
    lines.append(f"- 时间：{record['scan_time']}")
    lines.append("- 监控范围：NPU 0–4")
    marker = f"`{record['dedup_key']}`"
    if event_line is not None:
        marker = f"{marker}（审计行 {event_line}）"
    lines.append(f"- 事件：{marker}")
    lines.append(f"- 负责人：{record['owner']}")
    lines.append(f"- 恢复状态：{record['recovery_status']}")
    if record.get("npu4_unregistered_runner"):
        lines.append("- NPU 4 未注册 runner，任何占用一律视为违规。")
    lines.append("")
    lines.append("| NPU | PID | 判定 | 用户 | 进程 | 可执行文件 | 显存 | 处理结果 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    vram = _display(record.get("vram_mb"))
    if vram != "unknown":
        vram = f"{vram} MB"
    lines.append(
        "| {} | {} | `{}` | {} | {} | {} | {} | `{}` |".format(
            record["npu"],
            record["pid"],
            record["determination"],
            _display(record.get("user")),
            record["process"],
            _display(record.get("exe")),
            vram,
            record["result"],
        )
    )
    lines.append("")
    lines.append("命令行参数未上传，避免其中可能包含密钥。")
    lines.append(f"完整处置记录保存在宿主机 `{DEFAULT_EVENTS_PATH}`。")
    return "\n".join(lines)


def verify_summary_consistency(
    summary: str, record: Mapping[str, Any], event_line: int | None = None
) -> list[str]:
    """Re-parse a rendered summary and prove it matches the event record.

    Returns a list of mismatch messages (empty == consistent). This is the
    enforcement point for the "host JSONL audit <-> GitHub summary consistency"
    requirement: the daemon always posts ``render_github_summary`` output, and
    this function can also check a *previously* posted comment against the
    audit record it claims to represent.
    """
    mismatches: list[str] = []
    if event_line is not None:
        marker = f"（审计行 {event_line}）"
        if marker not in summary:
            mismatches.append(f"summary missing audit line marker {marker!r}")
    key = str(record["dedup_key"])
    if key not in summary:
        mismatches.append(f"summary does not mention dedup_key {key!r}")
    recovery = str(record["recovery_status"])
    if recovery not in summary:
        mismatches.append(f"summary does not mention recovery_status {recovery!r}")

    rows = [
        line
        for line in summary.splitlines()
        if line.startswith("|") and not line.startswith("|---")
    ]
    if len(rows) < 2:
        mismatches.append("summary has no data row")
        return mismatches
    cells = [cell.strip() for cell in rows[1].strip("|").split("|")]
    vram = _display(record.get("vram_mb"))
    if vram != "unknown":
        vram = f"{vram} MB"
    expected = {
        0: str(record["npu"]),
        1: str(record["pid"]),
        2: f"`{record['determination']}`",
        3: _display(record.get("user")),
        4: str(record["process"]),
        5: _display(record.get("exe")),
        6: vram,
        7: f"`{record['result']}`",
    }
    for index, want in expected.items():
        got = cells[index] if index < len(cells) else "<missing>"
        if got != want:
            mismatches.append(
                f"table column {index}: summary {got!r} != record {want!r}"
            )
    return mismatches


def parse_npu_smi_processes(text: str) -> list[dict[str, Any]]:
    """Parse ``npu-smi info -t process`` output into per-process rows.

    The exact ``npu-smi`` column layout varies across CANN versions, so the
    parser is intentionally tolerant: it looks for lines carrying an NPU index,
    a PID, a process name and a memory value, and skips header/separator lines.

    Each returned row contains at most ``npu``, ``pid``, ``process``,
    ``vram_mb`` and ``user`` (fields that cannot be determined are omitted).
    """
    rows: list[dict[str, Any]] = []
    line_pattern = re.compile(
        r"(?P<npu>\d+)\s+"
        r"(?P<pid>\d+)\s+"
        r"(?P<user>[^\s]+)\s+"
        r"(?P<process>\S+)\s+"
        r"(?P<vram>\d+)"
    )
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("-"):
            continue
        match = line_pattern.search(line)
        if match is None:
            continue
        npu = int(match.group("npu"))
        if npu < NPU_MIN or npu > NPU_MAX:
            continue
        rows.append(
            {
                "npu": npu,
                "pid": int(match.group("pid")),
                "user": match.group("user"),
                "process": match.group("process"),
                "vram_mb": int(match.group("vram")),
            }
        )
    return rows
