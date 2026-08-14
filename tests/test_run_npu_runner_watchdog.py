"""Tests for the deployable watchdog daemon (scripts/run_npu_runner_watchdog.py).

Covers the issue #125 acceptance item "one clean runner record plus one
violating-process record" via ``run_scan`` in dry-run mode: a runner-job /
sibling-container process must be left alone (action=none, result=no-op) while
an unauthorized container must be reclaimed (action=sigterm, result=terminated
in dry-run) and both must produce schema-conforming audit records.
"""

from __future__ import annotations

import json

from scripts.run_npu_runner_watchdog import (
    build_record,
    decide_action,
    load_facts_file,
    load_state,
    reclaim,
    run_scan,
    save_state,
)
from vllm_hust_benchmark.watchdog_ops import (
    derive_cmdline_sha256,
    derive_dedup_key,
    validate_event_record,
)


def _smi_text(*rows: tuple[int, int, str, str, int]) -> str:
    """Render fake ``npu-smi info -t process`` rows."""
    lines = ["NPU ID PID USER PROCESS MEMORY(MB)", "----- --- ---- ------- ----------"]
    for npu, pid, user, process, vram in rows:
        lines.append(f"{npu} {pid} {user} {process} {vram}")
    return "\n".join(lines)


def test_decide_action_none_for_owned_and_sigterm_for_violation() -> None:
    assert decide_action("runner-job") == "none"
    assert decide_action("sibling-container") == "none"
    assert decide_action("unauthorized-container") == "sigterm"
    assert decide_action("unowned-process") == "sigterm"


def test_reclaim_dry_run_returns_terminated() -> None:
    assert reclaim(999999, dry_run=True, sigkill_delay=0) == "terminated"


def test_build_record_redacts_cmdline_and_marks_npu4() -> None:
    record = build_record(
        {
            "npu": 4,
            "pid": 42,
            "user": "root",
            "process": "VLLMEngineCor",
            "exe": "/usr/bin/python3",
            "vram_mb": 111,
            "cmdline": ["python3", "serve.py", "--secret=abc"],
        },
        None,
        determination="unowned-process",
        action="sigterm",
        result="terminated",
        owner="SuccinctPaul",
        sequence=1,
        scan_epoch=1755000000,
        scan_time="2026-08-12T18:00:00+08:00",
        host="host-192-168-0-6",
    )
    assert record["cmdline_redacted"] is True
    assert record["cmdline_sha256"] == derive_cmdline_sha256(
        ["python3", "serve.py", "--secret=abc"]
    )
    assert record["cmdline_sha256"] != "abc"
    assert record["npu4_unregistered_runner"] is True
    assert record["dedup_key"] == derive_dedup_key(4, 42, record["cmdline_sha256"])
    assert validate_event_record(record) == []


def test_run_scan_clean_runner_plus_violating_process(tmp_path) -> None:
    """One clean runner record + one violating-process record (acceptance)."""
    npu_smi = _smi_text(
        (1, 1001, "root", "python3", 512),  # runner-job
        (2, 2002, "root", "VLLMEngineCor", 28345),  # unauthorized-container
    )
    facts = {
        "1001": {
            "container_name": "poy-180-21rc-npu1",
            "container_runner_label": "poy-180-21rc-npu1",
            "npu_physical_label": "1",
        },
        "2002": {
            "container_name": "rogue-job",
            "container_runner_label": None,
            "npu_physical_label": None,
        },
    }
    events = tmp_path / "events.jsonl"
    state_file = tmp_path / "state.json"

    appended = run_scan(
        npus=[1, 2],
        owner="SuccinctPaul",
        events=events,
        state_file=state_file,
        repo="vLLM-HUST/vllm-hust-benchmark",
        issue=125,
        gh_bin="gh",
        dry_run=True,
        sigkill_delay=0,
        host="host-192-168-0-6",
        sequence_start=1,
        npu_smi_text=npu_smi,
        facts_by_pid=facts,
    )
    assert appended == 2
    lines = events.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    by_pid = {record["pid"]: record for record in records}

    runner = by_pid[1001]
    assert runner["determination"] == "runner-job"
    assert runner["action"] == "none"
    assert runner["result"] == "no-op"
    assert runner["recovery_status"] == "open"
    assert runner["alert_suppressed"] is True  # owned process; audit-only, never alerts

    rogue = by_pid[2002]
    assert rogue["determination"] == "unauthorized-container"
    assert rogue["action"] == "sigterm"
    assert rogue["result"] == "terminated"  # dry-run reclaim
    assert rogue["alert_suppressed"] is False

    for record in records:
        assert validate_event_record(record) == []


def test_run_scan_dedup_suppresses_second_scan(tmp_path) -> None:
    """The same open event must not alert twice (dedup acceptance)."""
    npu_smi = _smi_text((4, 777, "root", "VLLMEngineCor", 999))
    facts = {
        "777": {
            "container_name": "rogue",
            "container_runner_label": None,
            "npu_physical_label": None,
        }
    }
    events = tmp_path / "events.jsonl"
    state_file = tmp_path / "state.json"

    first = run_scan(
        npus=[4],
        owner="SuccinctPaul",
        events=events,
        state_file=state_file,
        repo="vLLM-HUST/vllm-hust-benchmark",
        issue=125,
        gh_bin="gh",
        dry_run=True,
        sigkill_delay=0,
        host="host-192-168-0-6",
        sequence_start=1,
        npu_smi_text=npu_smi,
        facts_by_pid=facts,
    )
    run_scan(
        npus=[4],
        owner="SuccinctPaul",
        events=events,
        state_file=state_file,
        repo="vLLM-HUST/vllm-hust-benchmark",
        issue=125,
        gh_bin="gh",
        dry_run=True,
        sigkill_delay=0,
        host="host-192-168-0-6",
        sequence_start=1 + first,
        npu_smi_text=npu_smi,
        facts_by_pid=facts,
    )
    # Both scans append records, but the second must be alert-suppressed.
    lines = events.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    second_record = json.loads(lines[1])
    assert second_record["alert_suppressed"] is True
    assert second_record["event_sequence"] == 2


def test_load_save_state_round_trip(tmp_path) -> None:
    path = tmp_path / "state.json"
    assert load_state(path) == {}
    save_state(
        path, {"npu4/pid1/cmdx": {"result": "killed", "recovery_status": "open"}}
    )
    state = load_state(path)
    assert state["npu4/pid1/cmdx"]["result"] == "killed"
    # corrupted state must not crash
    path.write_text("{not json", encoding="utf-8")
    assert load_state(path) == {}


def test_load_facts_file_tolerates_bad_input(tmp_path) -> None:
    assert load_facts_file(None) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("nope", encoding="utf-8")
    assert load_facts_file(bad) == {}
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"123": {"container_name": "x"}}), encoding="utf-8")
    assert load_facts_file(good) == {"123": {"container_name": "x"}}
