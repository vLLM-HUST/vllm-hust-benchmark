import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run-current-ascend-same-spec.sh"


def _function_text(script: str, function_name: str) -> str:
    function_start = script.index(f"{function_name}()")
    function_end = script.index("\n}\n", function_start) + len("\n}\n")
    return script[function_start:function_end]


def _run_log_detector(function_name: str, tmp_path: Path, log_text: str) -> int:
    script = RUNNER.read_text(encoding="utf-8")
    detector = _function_text(script, function_name)
    log_file = tmp_path / "server.log"
    log_file.write_text(log_text, encoding="utf-8")

    result = subprocess.run(
        ["bash", "-c", f'{detector}\n{function_name} "$1"', "bash", str(log_file)],
        check=False,
    )
    return result.returncode


def test_current_same_spec_runner_preserves_wait_for_server_exit_code() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    serve_block = script[script.index('if [[ "$BENCHMARK_TYPE" == "serve" ]]; then') :]
    wait_block = serve_block[
        serve_block.index('wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"') :
    ]
    wait_block = wait_block[: wait_block.index('if [[ "$server_wait_status" -eq 86')]

    assert "set +e" in wait_block
    assert 'wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"' in wait_block
    assert "server_wait_status=$?" in wait_block
    assert "set -e" in wait_block
    assert 'if [[ "$server_wait_status" -eq 0 ]]; then' in wait_block


def test_current_same_spec_runner_does_not_retry_generic_engine_startup_wrapper() -> (
    None
):
    script = RUNNER.read_text(encoding="utf-8")

    detector = script[script.index("server_log_indicates_node_env_failure()") :]
    detector = detector[: detector.index("wait_for_server()")]

    assert "Engine core initialization failed" not in detector
    assert "ERR99999 UNKNOWN applicaiton exception" not in detector
    assert "ERR99999 UNKNOWN application exception" not in detector
    assert "rtGetDeviceCount" in detector
    assert "Resource_Busy" in detector


def test_current_same_spec_runner_detects_explicit_npu_oom_patterns(
    tmp_path: Path,
) -> None:
    messages = (
        "torch.OutOfMemoryError: NPU exhausted",
        "RuntimeError: NPU out of memory. Tried to allocate 2 GiB",
        "runtime failed: ACL_ERROR_RT_MEMORY_ALLOCATION",
        "Call aclrtMalloc failed, ret: 207001",
        "Failed to allocate NPU device memory",
    )

    for message in messages:
        assert _run_log_detector("server_log_indicates_npu_oom", tmp_path, message) == 0


def test_current_same_spec_runner_does_not_classify_generic_error_as_oom(
    tmp_path: Path,
) -> None:
    message = (
        "ERR99999 UNKNOWN applicaiton exception: Engine core initialization failed"
    )

    assert _run_log_detector("server_log_indicates_npu_oom", tmp_path, message) != 0
    assert (
        _run_log_detector("server_log_indicates_node_env_failure", tmp_path, message)
        != 0
    )


def test_current_same_spec_runner_keeps_resource_busy_retryable(tmp_path: Path) -> None:
    message = "RuntimeError: Resource_Busy(EL0005): The resources are busy"

    assert (
        _run_log_detector("server_log_indicates_node_env_failure", tmp_path, message)
        == 0
    )
    assert _run_log_detector("server_log_indicates_npu_oom", tmp_path, message) != 0


def test_current_same_spec_runner_treats_npu_oom_as_terminal() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    wait_block = script[script.index("wait_for_server()") :]
    wait_block = wait_block[: wait_block.index("resolved_dataset_path=")]
    retry_block = script[script.index("for start_attempt in") :]
    retry_block = retry_block[: retry_block.index("run_client_command")]

    assert "NPU_OOM_EXIT_CODE=87" in script
    assert wait_block.count('return "$NPU_OOM_EXIT_CODE"') >= 3
    assert wait_block.count("server_log_indicates_npu_oom") >= 3
    assert "refusing to retry server startup" in wait_block
    assert '"$server_wait_status" -eq 86' in retry_block
    assert '"$server_wait_status" -eq "$NPU_OOM_EXIT_CODE"' not in retry_block


def test_current_same_spec_runner_defaults_to_one_start_attempt() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "SERVER_START_RETRIES=${SERVER_START_RETRIES:-1}" in script


def test_current_same_spec_runner_requires_offline_graph_proof() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "VLLM_HUST_REQUIRE_OFFLINE_GRAPH=1" in script
    assert "VLLM_HUST_OFFLINE_GRAPH_PROOF_FILE" in script
    assert "graph_mode_verified == true" in script
    assert "enforce_eager == false" in script
    assert "CURRENT_ALLOW_OFFLINE_EAGER_BENCHMARK" not in script


def test_wait_for_server_fails_while_live_server_logs_npu_oom(
    tmp_path: Path,
) -> None:
    script = RUNNER.read_text(encoding="utf-8")
    detector = _function_text(script, "server_log_indicates_npu_oom")
    print_tail = _function_text(script, "print_server_log_tail")
    wait_for_server = _function_text(script, "wait_for_server")
    log_file = tmp_path / "server.log"
    log_file.write_text("starting\n", encoding="utf-8")

    harness = f"""#!/bin/bash
set -u
{detector}
{print_tail}
probe_server_ready() {{ return 1; }}
{wait_for_server}
NPU_OOM_EXIT_CODE=87
READY_TIMEOUT_SECONDS=100
READY_STATUS_INTERVAL_SECONDS=100
SERVER_LOG_PROGRESS_INTERVAL_SECONDS=0
SERVER_LOG_PROGRESS_TAIL_LINES=5
SERVER_LOG_OOM_CHECK_INTERVAL_SECONDS=1
SERVER_STDOUT_LOG="$1"
SERVER_PID=$$
sleep() {{ command sleep 0.01; }}
(command sleep 0.03; echo 'torch.OutOfMemoryError: NPU exhausted' >> "$1") &
wait_for_server 127.0.0.1 1
"""
    result = subprocess.run(
        ["bash", "-c", harness, "bash", str(log_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 87
    assert "out of memory while waiting" in result.stderr


def test_client_monitor_returns_dedicated_exit_code_for_npu_oom(
    tmp_path: Path,
) -> None:
    script = RUNNER.read_text(encoding="utf-8")
    detector = _function_text(script, "server_log_indicates_npu_oom")
    monitor = _function_text(script, "run_client_command_with_server_monitor")
    log_file = tmp_path / "server.log"
    log_file.write_text("ACL_ERROR_RT_MEMORY_ALLOCATION\n", encoding="utf-8")

    harness = f"""#!/bin/bash
set -u
{detector}
run_client_command() {{ command sleep 10; }}
cleanup_managed_server() {{ return 0; }}
{monitor}
NPU_OOM_EXIT_CODE=87
CLIENT_SERVER_MONITOR_INTERVAL_SECONDS=1
BENCHMARK_TYPE=serve
SERVER_STDOUT_LOG="$1"
run_client_command_with_server_monitor
"""
    result = subprocess.run(
        ["bash", "-c", harness, "bash", str(log_file)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 87
    assert "while running" in result.stderr


def test_current_same_spec_runner_prints_enough_server_log_context() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "SERVER_LOG_TAIL_LINES=${SERVER_LOG_TAIL_LINES:-200}" in script
    assert (
        "SERVER_LOG_PROGRESS_INTERVAL_SECONDS=${SERVER_LOG_PROGRESS_INTERVAL_SECONDS:-120}"
        in script
    )
    assert (
        "SERVER_LOG_PROGRESS_TAIL_LINES=${SERVER_LOG_PROGRESS_TAIL_LINES:-40}" in script
    )
    assert "print_server_log_tail()" in script
    assert 'print_server_log_tail "$SERVER_STDOUT_LOG"' in script
    assert "tail -n 40" not in script


def test_current_same_spec_runner_prints_server_log_progress_while_waiting() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    wait_block = script[script.index("wait_for_server()") :]
    wait_block = wait_block[: wait_block.index('  echo "Timed out waiting')]

    assert "next_log_progress_at" in wait_block
    assert "same-spec server log progress at" in wait_block
    assert 'probe_server_ready "$host" "$port" 1 || true' in wait_block
    assert (
        'print_server_log_tail "$SERVER_STDOUT_LOG" "$SERVER_LOG_PROGRESS_TAIL_LINES"'
        in wait_block
    )


def test_current_same_spec_runner_probe_bypasses_proxy_and_checks_ping() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    probe_block = script[script.index("probe_server_ready()") :]
    probe_block = probe_block[
        : probe_block.index("server_log_indicates_node_env_failure()")
    ]

    assert "READY_PROBE_TIMEOUT_SECONDS=${READY_PROBE_TIMEOUT_SECONDS:-5}" in script
    assert '"/ping"' in probe_block
    assert "socket.create_connection" in probe_block
    assert "set +e" in probe_block
    assert "set -e" in probe_block
    assert "readiness probe failed:" in probe_block
    assert "curl " not in probe_block
