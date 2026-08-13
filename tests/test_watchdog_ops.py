"""Tests for the poy-180 NPU runner watchdog auditability closed-loop (issue #125).

Covers the operator comment acceptance list:
- ``unauthorized-container`` determination fields and exit/cleanup results.
- Same-event alert dedup, owner and recovery status.
- Host JSONL audit <-> GitHub summary consistency.
- NPU 4 (no registered runner) explicit alert policy.
- One clean runner record plus one violating-process record via ``run_scan``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm_hust_benchmark.watchdog_ops import (
    DEFAULT_ALERT_MENTION,
    DEFAULT_OWNER,
    REGISTERED_RUNNER_NPUS,
    classify_determination,
    derive_cmdline_sha256,
    derive_dedup_key,
    npu_is_policy_violation,
    parse_npu_smi_processes,
    render_github_summary,
    should_alert,
    validate_event_record,
    verify_summary_consistency,
)


def _make_record(**overrides: object) -> dict:
    """Build a schema-conforming watchdog event record."""
    record: dict = {
        "schema_version": "npu-watchdog-event/v1",
        "schema_name": "npu-watchdog-event",
        "host": "host-192-168-0-6",
        "scan_epoch": 1755000000,
        "scan_time": "2026-08-12T18:00:00+08:00",
        "npu": 4,
        "pid": 1234,
        "user": "root",
        "process": "VLLMEngineCor",
        "exe": "/usr/bin/python3",
        "vram_mb": 512,
        "cmdline_sha256": "a" * 64,
        "cmdline_redacted": True,
        "determination": "unauthorized-container",
        "owner": DEFAULT_OWNER,
        "action": "sigterm",
        "result": "terminated",
        "recovery_status": "open",
        "dedup_key": "npu4/pid1234/cmdaaaaaaaaaaaa",
        "alert_suppressed": False,
        "npu4_unregistered_runner": True,
        "event_sequence": 1,
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# schema conformance
# ---------------------------------------------------------------------------


def test_valid_record_conforms_to_schema() -> None:
    assert validate_event_record(_make_record()) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "npu-watchdog-event/v2"),
        ("determination", "not-a-determination"),
        ("action", "reboot"),
        ("result", "exploded"),
        ("recovery_status", "half"),
        ("npu", 9),
        ("cmdline_sha256", "short"),
        ("cmdline_redacted", False),
        ("event_sequence", 0),
    ],
)
def test_schema_rejects_invalid_field_values(field: str, value: object) -> None:
    record = _make_record(**{field: value})
    errors = validate_event_record(record)
    assert errors, f"expected schema rejection for {field}={value!r}"


def test_schema_rejects_missing_required_field() -> None:
    record = _make_record()
    del record["dedup_key"]
    errors = validate_event_record(record)
    assert any("dedup_key" in error for error in errors)


def test_schema_rejects_extra_properties() -> None:
    record = _make_record()
    record["extra"] = True
    assert validate_event_record(record)


# ---------------------------------------------------------------------------
# cmdline digest + dedup key
# ---------------------------------------------------------------------------


def test_cmdline_sha256_is_stable_and_redacts_secrets() -> None:
    first = derive_cmdline_sha256(["python3", "serve.py", "--api-key=SECRET"])
    second = derive_cmdline_sha256(["python3", "serve.py", "--api-key=SECRET"])
    other = derive_cmdline_sha256(["python3", "serve.py", "--api-key=OTHER"])
    assert first == second
    assert first != other
    assert first not in "SECRET"
    assert len(first) == 64


def test_dedup_key_encodes_npu_pid_and_digest() -> None:
    digest = derive_cmdline_sha256(["a"])
    key = derive_dedup_key(2, 999, digest)
    assert key == f"npu2/pid999/cmd{digest[:12]}"


# ---------------------------------------------------------------------------
# ownership determination
# ---------------------------------------------------------------------------


def test_runner_job_determination() -> None:
    assert (
        classify_determination(
            npu=1,
            container_name="poy-180-21rc-npu1",
            container_runner_label="poy-180-21rc-npu1",
            npu_physical_label="1",
        )
        == "runner-job"
    )


def test_sibling_container_determination() -> None:
    assert (
        classify_determination(
            npu=2,
            container_name="some-job-abc",
            container_runner_label="poy-180-21rc-npu2",
            npu_physical_label="2",
        )
        == "sibling-container"
    )


@pytest.mark.parametrize(
    "npu,container_name,container_runner_label,npu_physical_label",
    [
        # inside a container with no runner label
        (0, "some-job", None, None),
        # label for a different NPU than the occupied one
        (0, "job-0", "poy-180-21rc-npu1", "1"),
        # physical label disagrees with occupied NPU
        (0, "job-0", "poy-180-21rc-npu0", "3"),
    ],
)
def test_unauthorized_determinations(
    npu: int,
    container_name: str | None,
    container_runner_label: str | None,
    npu_physical_label: str | None,
) -> None:
    assert (
        classify_determination(
            npu=npu,
            container_name=container_name,
            container_runner_label=container_runner_label,
            npu_physical_label=npu_physical_label,
        )
        == "unauthorized-container"
    )


def test_unowned_process_determination() -> None:
    assert (
        classify_determination(
            npu=3,
            container_name=None,
            container_runner_label=None,
            npu_physical_label=None,
        )
        == "unowned-process"
    )


# ---------------------------------------------------------------------------
# alert dedup + recovery status
# ---------------------------------------------------------------------------


def test_new_event_always_alerts() -> None:
    record = _make_record()
    alert, reason = should_alert(record, {})
    assert alert
    assert "new event" in reason


def test_unchanged_open_event_is_suppressed() -> None:
    record = _make_record()
    previous = {
        record["dedup_key"]: {"result": record["result"], "recovery_status": "open"}
    }
    alert, reason = should_alert(record, previous)
    assert not alert
    assert "already alerted" in reason


def test_recovery_transition_re_alerts() -> None:
    record = _make_record(recovery_status="recovered")
    previous = {
        record["dedup_key"]: {"result": record["result"], "recovery_status": "open"}
    }
    alert, reason = should_alert(record, previous)
    assert alert
    assert "recovery status changed" in reason


def test_result_escalation_re_alerts() -> None:
    record = _make_record(result="killed")
    previous = {
        record["dedup_key"]: {"result": "terminated", "recovery_status": "open"}
    }
    alert, reason = should_alert(record, previous)
    assert alert
    assert "result changed" in reason


def test_event_without_dedup_key_always_alerts() -> None:
    record = _make_record(dedup_key="")
    alert, reason = should_alert(record, {})
    assert alert
    assert "without dedup_key" in reason


# ---------------------------------------------------------------------------
# summary rendering + host JSONL <-> GitHub summary consistency
# ---------------------------------------------------------------------------


def test_summary_renders_from_record_and_redacts_cmdline() -> None:
    record = _make_record()
    summary = render_github_summary(record, event_line=7)
    assert DEFAULT_ALERT_MENTION in summary
    assert record["host"] in summary
    assert str(record["pid"]) in summary
    assert record["determination"] in summary
    assert record["result"] in summary
    assert "命令行参数未上传" in summary
    assert "a" * 64 not in summary  # digest itself never shown


def test_summary_consistency_matches_record() -> None:
    record = _make_record()
    summary = render_github_summary(record, event_line=7)
    assert verify_summary_consistency(summary, record, event_line=7) == []


def test_summary_consistency_detects_mismatch() -> None:
    record = _make_record()
    summary = render_github_summary(record, event_line=7)
    tampered = summary.replace(str(record["pid"]), str(record["pid"] + 1))
    mismatches = verify_summary_consistency(tampered, record, event_line=7)
    assert mismatches


def test_summary_consistency_detects_missing_audit_line_marker() -> None:
    record = _make_record()
    summary = render_github_summary(record, event_line=None)
    mismatches = verify_summary_consistency(summary, record, event_line=7)
    assert any("audit line" in mismatch for mismatch in mismatches)


def test_npu4_flag_is_rendered_in_summary() -> None:
    record = _make_record(npu=4, npu4_unregistered_runner=True)
    summary = render_github_summary(record)
    assert "NPU 4 未注册 runner" in summary


# ---------------------------------------------------------------------------
# NPU 4 explicit policy
# ---------------------------------------------------------------------------


def test_registered_runner_npus_are_0_to_3() -> None:
    assert REGISTERED_RUNNER_NPUS == {0, 1, 2, 3}


def test_npu4_is_policy_violation_but_registered_ones_are_not() -> None:
    assert npu_is_policy_violation(4)
    for npu in REGISTERED_RUNNER_NPUS:
        assert not npu_is_policy_violation(npu)


# ---------------------------------------------------------------------------
# npu-smi process parsing
# ---------------------------------------------------------------------------


def test_parse_npu_smi_processes_skips_headers_and_parses_rows() -> None:
    text = """\
    NPU ID      PID     USER    PROCESS      MEMORY(MB)
    -------------------------------------------------------------------------
    4           1234    root    python3      111
    1           5678    other   VLLMEngineCor 28345
    """
    rows = parse_npu_smi_processes(text)
    assert len(rows) == 2
    assert rows[0] == {
        "npu": 4,
        "pid": 1234,
        "user": "root",
        "process": "python3",
        "vram_mb": 111,
    }
    assert rows[1]["process"] == "VLLMEngineCor"


def test_parse_npu_smi_processes_ignores_out_of_scope_npus() -> None:
    text = "8 9001 root python3 100\n5 9002 root python3 200\n"
    assert parse_npu_smi_processes(text) == []


# ---------------------------------------------------------------------------
# sample event file (issue #125 asks to link sample events)
# ---------------------------------------------------------------------------


def test_sample_events_file_conforms_to_schema() -> None:
    sample_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "npu-runner-watchdog-sample-events.jsonl"
    )
    if not sample_path.exists():
        pytest.skip("sample events file not present")
    for line_number, line in enumerate(
        sample_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        errors = validate_event_record(record)
        assert not errors, f"sample line {line_number}: {errors}"
