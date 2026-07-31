"""Tests for issue #97 failure-diagnostics and defensive-cleanup hardening.

Covers ``capture_failure_diagnostics``, ``force_cleanup_managed_server``,
``verify_cleanup_success``, ``verify_npu_idle``, ``parse_npu_smi_processes``,
and the ``_HealthTimeoutError`` exception that carries a ``blocked_dir``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backfill_historical_pr_benchmarks.py"
)


@pytest.fixture(scope="module")
def module():
    spec = importlib.util.spec_from_file_location("historical_pr_backfill", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def target(module):
    return module.TargetRef(
        label="pr80-test",
        core_ref="ae16d09435",  # pragma: allowlist secret
        plugin_ref="a05a9efe54",
        pr_number=80,
    )


@pytest.fixture()
def spec(module):
    return module.OfficialSpec(
        path=Path("/tmp/official-prefix-repetition.json"),
        scenario="prefix-repetition-online",
        workload="prefix-repetition-online",
        benchmark_type="serve",
        model="Qwen/Qwen2.5-14B-Instruct",
        precision="FP16",
        chip_model="910B2",
        chip_count=1,
        node_count=1,
    )


@pytest.fixture()
def args(module):
    return argparse.Namespace(
        managed_container="vllm-hust-test",
        managed_systemd_unit="vllm-hust-test.service",
        managed_npu_devices="0",
        server_port="8420",
        dev_hub_dir="/tmp/dev-hub",
        allow_busy_npu=False,
        skip_defensive_cleanup=False,
    )


NPU_SMI_WITH_PROCESS = """\
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.0.rc1                            Version: 26.0.rc1                                                    |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B2               | OK            | 424.0                64                      0    / 0                |
| 0                         | 0000:C1:00.0  | 100                  0    / 0                38711/ 65536            |
+===========================+===============+======================================================================+
| 1     910B2               | OK            | 93.6                 44                      0    / 0                |
| 0                         | 0000:01:00.0  | 0                    0    / 0                3406 / 65536            |
+===========================+===============+======================================================================+
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
| 0       0                 | 1504884       |                    | 35358                 | 412309                  |
+===========================+===============+======================================================================+
| No running processes found in NPU 1                                                                              |
+===========================+===============+======================================================================+
"""

NPU_SMI_IDLE = """\
+------------------------------------------------------------------------------------------------------------------+
| npu-smi 26.0.rc1                            Version: 26.0.rc1                                                    |
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU   Name                | Health        | Power(W)             Temp(C)                 Hugepages-Usage(page)   |
| Chip                      | Bus-Id        | AICore(%)            Memory-Usage(MB)        HBM-Usage(MB)           |
+===========================+===============+======================================================================+
| 0     910B2               | OK            | 424.0                64                      0    / 0                |
| 0                         | 0000:C1:00.0  | 100                  0    / 0                38711/ 65536            |
+===========================+===============+======================================================================+
+---------------------------+---------------+----------------------------------------------------------------------+
| NPU     Chip              | Process id    | Process name       | Process memory(MB)    | Process id in container |
+===========================+===============+======================================================================+
| No running processes found in NPU 0                                                                              |
+===========================+===============+======================================================================+
"""


# === parse_npu_smi_processes ===


def test_parse_npu_smi_processes_extracts_device_pid_pairs(module):
    result = module.parse_npu_smi_processes(NPU_SMI_WITH_PROCESS, "0")
    assert result == [(0, 1504884)]


def test_parse_npu_smi_processes_returns_empty_for_idle_npu(module):
    result = module.parse_npu_smi_processes(NPU_SMI_IDLE, "0")
    assert result == []


def test_parse_npu_smi_processes_filters_to_managed_devices(module):
    result = module.parse_npu_smi_processes(NPU_SMI_WITH_PROCESS, "1")
    assert result == []


# === verify_npu_idle ===


def test_verify_npu_idle_raises_when_occupied(module, monkeypatch):
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **kw: NPU_SMI_WITH_PROCESS
    )
    with pytest.raises(RuntimeError, match="NPU devices 0 not idle"):
        module.verify_npu_idle(devices="0", allow_busy=False, execute=True)


def test_verify_npu_idle_passes_when_idle(module, monkeypatch):
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **kw: NPU_SMI_IDLE
    )
    module.verify_npu_idle(devices="0", allow_busy=False, execute=True)


def test_verify_npu_idle_bypassed_with_allow_busy(module, monkeypatch):
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **kw: NPU_SMI_WITH_PROCESS
    )
    module.verify_npu_idle(devices="0", allow_busy=True, execute=True)


def test_verify_npu_idle_skipped_in_dry_run(module, monkeypatch):
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *a, **kw: NPU_SMI_WITH_PROCESS
    )
    module.verify_npu_idle(devices="0", allow_busy=False, execute=False)


# === _HealthTimeoutError ===


def test_health_timeout_error_carries_blocked_dir(module, tmp_path):
    blocked = tmp_path / "blocked"
    err = module._HealthTimeoutError("timeout", blocked_dir=blocked)
    assert err.blocked_dir == blocked
    assert "timeout" in str(err)


def test_health_timeout_error_blocked_dir_defaults_none(module):
    err = module._HealthTimeoutError("timeout")
    assert err.blocked_dir is None


# === capture_failure_diagnostics ===


def test_capture_failure_diagnostics_writes_blocked_txt(
    module, target, spec, args, tmp_path
):
    blocked_dir = module.capture_failure_diagnostics(
        target=target,
        spec=spec,
        args=args,
        error="test failure",
        result_root=tmp_path,
        run_key="test-key",
        execute=False,
    )
    blocked_txt = blocked_dir / module.BLOCKED_MARKER_FILE
    assert blocked_txt.exists()
    content = blocked_txt.read_text(encoding="utf-8")
    assert "blocked_at:" in content
    assert "run_key: test-key" in content
    assert "target_label: pr80-test" in content
    assert "core_ref: ae16d09435" in content
    assert "blocker_reason: test failure" in content
    assert "Do NOT upload this directory as a submission artifact" in content


def test_capture_failure_diagnostics_blocked_dir_under_benchmarks_not_submissions(
    module, target, spec, args, tmp_path
):
    """Issue #97 item 4: BLOCKED.txt must NEVER live under submissions/."""
    blocked_dir = module.capture_failure_diagnostics(
        target=target,
        spec=spec,
        args=args,
        error="err",
        result_root=tmp_path,
        run_key="k",
        execute=False,
    )
    blocked_str = str(blocked_dir)
    assert "submissions" not in blocked_str
    assert "blocked" in blocked_str


def test_capture_failure_diagnostics_is_idempotent(
    module, target, spec, args, tmp_path
):
    """Calling twice for the same run_key doesn't fail."""
    for _ in range(2):
        module.capture_failure_diagnostics(
            target=target,
            spec=spec,
            args=args,
            error="err",
            result_root=tmp_path,
            run_key="dup",
            execute=False,
        )
    blocked_txt = (
        tmp_path / module.BLOCKED_DIR_NAME / "dup" / module.BLOCKED_MARKER_FILE
    )
    assert blocked_txt.exists()


def test_capture_failure_diagnostics_copies_state_file(
    module, target, spec, args, tmp_path
):
    state_file = tmp_path / "state.json"
    state_file.write_text('{"runs": {}}', encoding="utf-8")
    blocked_dir = module.capture_failure_diagnostics(
        target=target,
        spec=spec,
        args=args,
        error="err",
        result_root=tmp_path,
        run_key="k",
        execute=False,
        state_file=state_file,
    )
    assert (blocked_dir / "state.json").exists()
    assert (blocked_dir / "state.json").read_text(encoding="utf-8") == '{"runs": {}}'


def test_capture_failure_diagnostics_does_not_raise_on_subprocess_failure(
    module, target, spec, args, tmp_path, monkeypatch
):
    """Issue #97: diagnostics capture must never mask the original error."""

    def failing_run(*a, **kw):
        raise OSError("simulated failure")

    monkeypatch.setattr(module.subprocess, "run", failing_run)
    blocked_dir = module.capture_failure_diagnostics(
        target=target,
        spec=spec,
        args=args,
        error="original error",
        result_root=tmp_path,
        run_key="k",
        execute=True,
    )
    assert (blocked_dir / module.BLOCKED_MARKER_FILE).exists()
    assert (
        (blocked_dir / "npu-smi.txt")
        .read_text(encoding="utf-8")
        .startswith("capture failed:")
    )


def test_capture_failure_diagnostics_captures_npu_smi_snapshot(
    module, target, spec, args, tmp_path, monkeypatch
):
    def fake_run(cmd, **kw):
        result = type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()
        if "npu-smi" in cmd:
            result.stdout = NPU_SMI_WITH_PROCESS
        return result

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    blocked_dir = module.capture_failure_diagnostics(
        target=target,
        spec=spec,
        args=args,
        error="err",
        result_root=tmp_path,
        run_key="k",
        execute=True,
    )
    npu_snapshot = (blocked_dir / "npu-smi.txt").read_text(encoding="utf-8")
    assert "910B2" in npu_snapshot
    assert "1504884" in npu_snapshot


# === force_cleanup_managed_server ===


def test_force_cleanup_kills_npu_processes(module, args, monkeypatch):
    monkeypatch.setattr(module, "detect_npu_processes", lambda devices: [(0, 99999)])
    killed: list[int] = []
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: killed.append(pid))

    report = module.force_cleanup_managed_server(args=args, execute=True)

    assert 99999 in killed
    assert 99999 in report["npu_procs_killed"]


def test_force_cleanup_stops_systemd_unit(module, args, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(module, "detect_npu_processes", lambda devices: [])
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.force_cleanup_managed_server(args=args, execute=True)

    systemctl_calls = [c for c in calls if "systemctl" in c]
    stop_calls = [c for c in systemctl_calls if "stop" in c]
    reset_calls = [c for c in systemctl_calls if "reset-failed" in c]
    assert len(stop_calls) == 1
    assert "vllm-hust-test.service" in stop_calls[0]
    assert len(reset_calls) == 1


def test_force_cleanup_frees_port(module, args, monkeypatch):
    def fake_run(cmd, **kw):
        if isinstance(cmd, list) and "lsof" in " ".join(cmd):
            return type("R", (), {"stdout": "12345\n", "stderr": "", "returncode": 0})()
        if isinstance(cmd, str) and "lsof" in cmd:
            return type("R", (), {"stdout": "12345\n", "stderr": "", "returncode": 0})()
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    killed: list[int] = []
    monkeypatch.setattr(module, "detect_npu_processes", lambda devices: [])
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: killed.append(pid))

    report = module.force_cleanup_managed_server(args=args, execute=True)

    assert 12345 in killed
    assert report["port_freed"] is True


def test_force_cleanup_removes_docker_container(module, args, monkeypatch):
    def fake_run(cmd, **kw):
        if "docker" in cmd and "rm" in cmd:
            return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(module, "detect_npu_processes", lambda devices: [])
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    report = module.force_cleanup_managed_server(args=args, execute=True)

    assert report["container_removed"] is True


def test_force_cleanup_returns_empty_report_in_dry_run(module, args):
    report = module.force_cleanup_managed_server(args=args, execute=False)
    assert report["container_removed"] is False
    assert report["systemd_stopped"] is False
    assert report["npu_procs_killed"] == []
    assert report["port_freed"] is False


def test_force_cleanup_writes_cleanup_log_to_blocked_dir(
    module, args, tmp_path, monkeypatch
):
    blocked_dir = tmp_path / "blocked" / "k"
    blocked_dir.mkdir(parents=True)
    monkeypatch.setattr(module, "detect_npu_processes", lambda devices: [(0, 99999)])
    monkeypatch.setattr(module.os, "kill", lambda pid, sig: None)

    module.force_cleanup_managed_server(
        args=args, execute=True, blocked_dir=blocked_dir
    )

    log = (blocked_dir / "cleanup.log").read_text(encoding="utf-8")
    assert "killed NPU process 99999" in log


# === verify_cleanup_success ===


def test_verify_cleanup_success_returns_warnings_for_npu_residuals(
    module, args, monkeypatch
):
    monkeypatch.setattr(module, "detect_npu_processes", lambda devices: [(0, 88888)])
    warnings = module.verify_cleanup_success(args=args, execute=True)
    assert any("residual NPU processes" in w for w in warnings)
    assert any("88888" in w for w in warnings)


def test_verify_cleanup_success_returns_empty_when_clean(module, args, monkeypatch):
    monkeypatch.setattr(module, "detect_npu_processes", lambda devices: [])

    def fake_run(cmd, **kw):
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    warnings = module.verify_cleanup_success(args=args, execute=True)
    assert warnings == []


def test_verify_cleanup_success_returns_empty_in_dry_run(module, args):
    warnings = module.verify_cleanup_success(args=args, execute=False)
    assert warnings == []


def test_verify_cleanup_success_warns_on_active_systemd(module, args, monkeypatch):
    def fake_run(cmd, **kw):
        if "is-active" in cmd:
            return type(
                "R", (), {"stdout": "active\n", "stderr": "", "returncode": 0}
            )()
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(module, "detect_npu_processes", lambda devices: [])
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    warnings = module.verify_cleanup_success(args=args, execute=True)
    assert any("vllm-hust-test.service" in w and "active" in w for w in warnings)


def test_verify_cleanup_success_warns_on_port_in_use(module, args, monkeypatch):
    def fake_run(cmd, **kw):
        if isinstance(cmd, list) and "lsof" in " ".join(cmd):
            return type("R", (), {"stdout": "54321\n", "stderr": "", "returncode": 0})()
        if isinstance(cmd, str) and "lsof" in cmd:
            return type("R", (), {"stdout": "54321\n", "stderr": "", "returncode": 0})()
        return type("R", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(module, "detect_npu_processes", lambda devices: [])
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    warnings = module.verify_cleanup_success(args=args, execute=True)
    assert any("8420" in w and "54321" in w for w in warnings)


# === CLI flags ===


def test_cli_flags_exist(module, monkeypatch):
    """Issue #97: --allow-busy-npu and --skip-defensive-cleanup must be available."""
    monkeypatch.setattr(sys, "argv", ["backfill_historical_pr_benchmarks.py"])
    args = module.parse_args()
    assert args.allow_busy_npu is False
    assert args.skip_defensive_cleanup is False


# === to_container_workspace_path (issue #97: env-var-driven host workspace root) ===


def test_to_container_workspace_path_defaults_to_home_shuhao(module, monkeypatch):
    monkeypatch.delenv("VLLM_HUST_HOST_WORKSPACE_ROOT", raising=False)
    monkeypatch.delenv("VLLM_HUST_CONTAINER_WORKSPACE_ROOT", raising=False)
    result = module.to_container_workspace_path(Path("/home/shuhao/vllm-hust"))
    assert result == "/workspace/vllm-hust"


def test_to_container_workspace_path_uses_env_var_for_custom_root(module, monkeypatch):
    monkeypatch.setenv("VLLM_HUST_HOST_WORKSPACE_ROOT", "/root/vllm")
    monkeypatch.delenv("VLLM_HUST_CONTAINER_WORKSPACE_ROOT", raising=False)
    result = module.to_container_workspace_path(Path("/root/vllm/vllm-hust"))
    assert result == "/workspace/vllm-hust"


def test_to_container_workspace_path_returns_raw_path_when_outside_root(
    module, monkeypatch
):
    monkeypatch.setenv("VLLM_HUST_HOST_WORKSPACE_ROOT", "/root/vllm")
    result = module.to_container_workspace_path(Path("/opt/other/repo"))
    assert result == "/opt/other/repo"


def test_require_container_workspace_path_raises_when_outside_root(module, monkeypatch):
    monkeypatch.setenv("VLLM_HUST_HOST_WORKSPACE_ROOT", "/root/vllm")
    with pytest.raises(RuntimeError, match="does not map into the dev-hub"):
        module.require_container_workspace_path(
            Path("/opt/other/repo"), purpose="core worktree"
        )


def test_require_container_workspace_path_in_container_returns_real_path(module):
    """Issue #97: in-container mode must use the real filesystem path."""
    real = Path("/root/vllm/vllm-hust-benchmark/.benchmarks/worktrees/core")
    result = module.require_container_workspace_path(
        real, purpose="core worktree", in_container=True
    )
    assert result == str(real.resolve())
    assert not result.startswith("/workspace/")


def test_require_container_workspace_path_in_container_skips_workspace_check(
    module, monkeypatch
):
    """in_container=True bypasses the /workspace/ prefix requirement."""
    monkeypatch.setenv("VLLM_HUST_HOST_WORKSPACE_ROOT", "/home/shuhao")
    result = module.require_container_workspace_path(
        Path("/opt/arbitrary/path"), purpose="plugin worktree", in_container=True
    )
    assert result == "/opt/arbitrary/path"


# === resolve_manage_script (issue #97: in-container launcher support) ===


def test_resolve_manage_script_defaults_to_manage_sh(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["backfill_historical_pr_benchmarks.py"])
    args = module.parse_args()
    args.dev_hub_dir = "/home/shuhao/vllm-hust-dev-hub"
    result = module.resolve_manage_script(args)
    assert result.name == "manage.sh"
    assert result.parent == Path("/home/shuhao/vllm-hust-dev-hub").resolve()


def test_resolve_manage_script_supports_container_launcher(module, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "backfill_historical_pr_benchmarks.py",
            "--manage-script",
            "scripts/manage-container.sh",
            "--dev-hub-dir",
            "/root/vllm/vllm-hust-dev-hub",
        ],
    )
    args = module.parse_args()
    result = module.resolve_manage_script(args)
    assert result.name == "manage-container.sh"
    assert result.parent.name == "scripts"
    assert result.parent.parent.name == "vllm-hust-dev-hub"


def test_manage_script_flag_defaults_to_manage_sh(module, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["backfill_historical_pr_benchmarks.py"])
    args = module.parse_args()
    assert args.manage_script == "manage.sh"


# === find_local_model_path env var override (issue #97) ===


def test_find_local_model_path_uses_env_var_override(module, monkeypatch, tmp_path):
    monkeypatch.delenv("VLLM_HUST_LOCAL_MODEL_PATH", raising=False)
    custom = tmp_path / "my-model"
    custom.mkdir()
    monkeypatch.setenv("VLLM_HUST_LOCAL_MODEL_PATH", str(custom))
    result = module.find_local_model_path("Qwen/Qwen2.5-14B-Instruct")
    assert result == str(custom)


def test_find_local_model_path_env_var_returns_none_when_path_missing(
    module, monkeypatch
):
    monkeypatch.setenv("VLLM_HUST_LOCAL_MODEL_PATH", "/nonexistent/path/12345")
    result = module.find_local_model_path("Qwen/Qwen2.5-14B-Instruct")
    assert result is None
