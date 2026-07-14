from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts/run-current-ascend-same-spec.sh"


def test_current_same_spec_runner_preserves_wait_for_server_exit_code() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    serve_block = script[script.index('if [[ "$BENCHMARK_TYPE" == "serve" ]]; then') :]
    wait_block = serve_block[serve_block.index('wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"') :]
    wait_block = wait_block[: wait_block.index('if [[ "$server_wait_status" -eq 86')]

    assert 'set +e' in wait_block
    assert 'wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"' in wait_block
    assert 'server_wait_status=$?' in wait_block
    assert 'set -e' in wait_block
    assert 'if [[ "$server_wait_status" -eq 0 ]]; then' in wait_block


def test_current_same_spec_runner_does_not_retry_generic_engine_startup_wrapper() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    detector = script[script.index("server_log_indicates_node_env_failure()") :]
    detector = detector[: detector.index("wait_for_server()")]

    assert "Engine core initialization failed" not in detector
    assert "ERR99999 UNKNOWN applicaiton exception" not in detector
    assert "ERR99999 UNKNOWN application exception" not in detector
    assert "rtGetDeviceCount" in detector
    assert "Resource_Busy" in detector


def test_current_same_spec_runner_prints_enough_server_log_context() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    assert "SERVER_LOG_TAIL_LINES=${SERVER_LOG_TAIL_LINES:-200}" in script
    assert "SERVER_LOG_PROGRESS_INTERVAL_SECONDS=${SERVER_LOG_PROGRESS_INTERVAL_SECONDS:-120}" in script
    assert "SERVER_LOG_PROGRESS_TAIL_LINES=${SERVER_LOG_PROGRESS_TAIL_LINES:-40}" in script
    assert "print_server_log_tail()" in script
    assert 'print_server_log_tail "$SERVER_STDOUT_LOG"' in script
    assert "tail -n 40" not in script


def test_current_same_spec_runner_prints_server_log_progress_while_waiting() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    wait_block = script[script.index("wait_for_server()") :]
    wait_block = wait_block[: wait_block.index("  echo \"Timed out waiting")]

    assert "next_log_progress_at" in wait_block
    assert "same-spec server log progress at" in wait_block
    assert 'probe_server_ready "$host" "$port" 1 || true' in wait_block
    assert 'print_server_log_tail "$SERVER_STDOUT_LOG" "$SERVER_LOG_PROGRESS_TAIL_LINES"' in wait_block


def test_current_same_spec_runner_probe_bypasses_proxy_and_checks_ping() -> None:
    script = RUNNER.read_text(encoding="utf-8")

    probe_block = script[script.index("probe_server_ready()") :]
    probe_block = probe_block[: probe_block.index("server_log_indicates_node_env_failure()")]

    assert "READY_PROBE_TIMEOUT_SECONDS=${READY_PROBE_TIMEOUT_SECONDS:-5}" in script
    assert '"/ping"' in probe_block
    assert "socket.create_connection" in probe_block
    assert "set +e" in probe_block
    assert "set -e" in probe_block
    assert "readiness probe failed:" in probe_block
    assert "curl " not in probe_block
