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
    assert "rtGetDeviceCount" in detector
    assert "Resource_Busy" in detector
