import os
import hashlib
import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from tests._bash_utils import bash_executable


def _spawn_process_tree(tmp_path: Path) -> tuple[subprocess.Popen[str], int]:
    """Spawn a wrapper script that runs a background sleep process.

    Avoid ``exec -a`` because it is fragile in sandboxed environments
    (the argv[0] rename can interfere with process-group creation).
    """
    child_pid_file = tmp_path / "child.pid"
    wrapper_script = tmp_path / "wrapper.sh"
    wrapper_script.write_text(
        "#!/bin/bash\n"
        f"sleep 300 &\n"
        f"echo $! > {shlex.quote(str(child_pid_file))}\n"
        "wait\n",
        encoding="utf-8",
    )
    wrapper_script.chmod(0o755)

    process = subprocess.Popen(
        [bash_executable(), str(wrapper_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        preexec_fn=os.setpgrp,
    )
    child_pid = _wait_for_pid_file(child_pid_file)
    return process, child_pid


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "scripts/prepare-official-ascend-baseline-env.sh"
RUN_OFFICIAL_SCRIPT = REPO_ROOT / "scripts/run-official-ascend-goal-baseline.sh"


def test_admission_only_preflight_does_not_require_conda() -> None:
    source = PREPARE_SCRIPT.read_text(encoding="utf-8")

    admission_branch = source.index(
        'if [[ "$PREPARE_BENCHMARK_ADMISSION_ONLY" == "1" ]]'
    )
    conda_requirement = source.index("if ! command -v conda")

    assert admission_branch < conda_requirement


def _source_prepare_functions(snippet: str) -> str:
    script_path = shlex.quote(str(PREPARE_SCRIPT))
    return (
        "source <(awk '/^if ! command -v conda / {exit} {print}' "
        f"{script_path}) && {snippet}"
    )


def _source_run_official_functions(snippet: str) -> str:
    script_path = shlex.quote(str(RUN_OFFICIAL_SCRIPT))
    return (
        r"source <(awk 'BEGIN{capture=0} /^set_ascend_visible_devices_scope\(\) \{/ {capture=1} /^run_server_command\(\) \{/ {exit} capture {print}' "
        f"{script_path}) && {snippet}"
    )


def _source_run_client_functions(snippet: str) -> str:
    script_path = shlex.quote(str(RUN_OFFICIAL_SCRIPT))
    return (
        r"source <(awk 'BEGIN{capture=0} /^run_client_command\(\) \{/ {capture=1} /^resolve_same_spec\(\) \{/ {exit} capture {print}' "
        f"{script_path}) && {snippet}"
    )


def _source_run_official_version_functions(snippet: str) -> str:
    script_path = shlex.quote(str(RUN_OFFICIAL_SCRIPT))
    return (
        r"source <(awk 'BEGIN{capture=0} /^normalize_engine_version\(\) \{/ {capture=1} /^kill_server\(\) \{/ {exit} capture {print}' "
        f"{script_path}) && {snippet}"
    )


def _source_run_official_runtime_model_functions(snippet: str) -> str:
    script_path = shlex.quote(str(RUN_OFFICIAL_SCRIPT))
    return (
        r"source <(awk 'BEGIN{capture=0} /^normalized_server_parameters_json\(\) \{/ {capture=1} /^kill_server\(\) \{/ {exit} capture {print}' "
        f"{script_path}) && {snippet}"
    )


def _source_worktree_functions(snippet: str) -> str:
    script_path = shlex.quote(str(RUN_OFFICIAL_SCRIPT))
    return (
        r"source <(awk 'BEGIN{capture=0} /^ensure_worktree\(\) \{/ {capture=1} /^json2args\(\) \{/ {exit} capture {print}' "
        f"{script_path}) && {snippet}"
    )


def _run_bash(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [bash_executable(), "-lc", command],
        check=check,
        capture_output=True,
        text=True,
    )


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_pid_file(pid_file: Path, timeout_seconds: float = 5.0) -> int:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if pid_file.exists():
            contents = pid_file.read_text(encoding="utf-8").strip()
            if contents:
                return int(contents)
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for child pid file: {pid_file}")


def _wait_for_pid_exit(pid: int, timeout_seconds: float = 5.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not _pid_exists(pid):
            return
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for pid {pid} to exit")


def _cleanup_process_tree(root_pid: int, child_pid: int | None) -> None:
    try:
        os.killpg(root_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    if child_pid is not None:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_collect_process_tree_pids_includes_child_process(tmp_path: Path) -> None:
    process, child_pid = _spawn_process_tree(tmp_path)

    try:
        result = _run_bash(
            _source_prepare_functions(f"collect_process_tree_pids {process.pid}")
        )
        collected_pids = {
            int(line) for line in result.stdout.splitlines() if line.strip()
        }

        assert process.pid in collected_pids
        assert child_pid in collected_pids
    finally:
        _cleanup_process_tree(process.pid, child_pid)


def test_terminate_pid_tree_kills_root_and_child(tmp_path: Path) -> None:
    process, child_pid = _spawn_process_tree(tmp_path)

    try:
        result = _run_bash(
            _source_prepare_functions(
                f'terminate_pid_tree {process.pid} "test process tree"'
            )
        )

        assert "[official-env] stopping test process tree:" in result.stdout
        process.wait(timeout=5)
        _wait_for_pid_exit(child_pid)
        assert not _pid_exists(process.pid)
        assert not _pid_exists(child_pid)
    finally:
        _cleanup_process_tree(process.pid, child_pid)


def test_is_process_in_cleanup_scope_requires_same_user_and_namespaces() -> None:
    result = _run_bash(
        _source_prepare_functions(
            """
                        CURRENT_PREPARE_USER_ID=1000
                        CURRENT_PREPARE_PID_NAMESPACE='pid:[11]'
                        CURRENT_PREPARE_MOUNT_NAMESPACE='mnt:[22]'

                        process_user_id() {
                            case "$1" in
                                101|102|103) printf '1000\n' ;;
                                104) printf '2000\n' ;;
                            esac
                        }

                        process_namespace() {
                            case "$1:$2" in
                                101:pid) printf 'pid:[11]\n' ;;
                                101:mnt) printf 'mnt:[22]\n' ;;
                                102:pid) printf 'pid:[11]\n' ;;
                                102:mnt) printf 'mnt:[33]\n' ;;
                                103:pid) printf 'pid:[99]\n' ;;
                                103:mnt) printf 'mnt:[22]\n' ;;
                                104:pid) printf 'pid:[11]\n' ;;
                                104:mnt) printf 'mnt:[22]\n' ;;
                            esac
                        }

                        for pid in 101 102 103 104; do
                            if is_process_in_cleanup_scope "$pid"; then
                                echo "allow:$pid"
                            else
                                echo "deny:$pid"
                            fi
                        done
                        """
        )
    )

    assert result.stdout.splitlines() == [
        "allow:101",
        "deny:102",
        "deny:103",
        "deny:104",
    ]


def test_residual_pid_lists_keep_only_in_scope_targets() -> None:
    result = _run_bash(
        _source_prepare_functions(
            """
                        # Mock all helper functions to ensure test isolation
                        process_user_id() {
                            echo "$CURRENT_PREPARE_USER_ID"
                        }

                        process_namespace() {
                            if [[ "$2" == "pid" ]]; then
                                echo "$CURRENT_PREPARE_PID_NAMESPACE"
                            elif [[ "$2" == "mnt" ]]; then
                                echo "$CURRENT_PREPARE_MOUNT_NAMESPACE"
                            fi
                        }

                        process_args() {
                            # Return empty to ensure no port conflicts
                            echo ""
                        }

                        list_managed_runtime_state_pids() {
                            printf '501\n502\n'
                        }

                        list_matching_benchmark_pids() {
                            printf '601\n602\n'
                        }

                        is_zombie_process() {
                            return 1
                        }

                        is_process_in_cleanup_scope() {
                            [[ "$1" == '501' ]]
                        }

                        is_benchmark_process() {
                            [[ "$1" == '601' || "$1" == '602' ]]
                        }

                        is_managed_runner_wrapper_process() {
                            [[ "$1" == '501' ]]
                        }

                        is_process_conflicting_with_benchmark_port() {
                            return 1
                        }

                        echo 'residual:'
                        list_benchmark_residual_pids
                        echo 'out-of-scope:'
                        list_out_of_scope_benchmark_pids
                        """
        )
    )

    assert result.stdout.splitlines() == [
        "residual:",
        "501",
        "out-of-scope:",
    ]


def test_list_matching_benchmark_pids_matches_cli_compat_process() -> None:
    result = _run_bash(
        _source_prepare_functions(
            """
            ps() {
                cat <<'EOF'
101 python /tmp/run_vllm_cli_compat.py bench serve --model foo
102 python -m vllm.entrypoints.cli.main bench serve --model foo
103 python /tmp/other.py serve --model foo
EOF
            }

            list_matching_benchmark_pids
            """
        )
    )

    assert result.stdout.splitlines() == ["101", "102"]


def test_run_in_official_env_python_uses_temp_script(tmp_path: Path) -> None:
    captured_args = tmp_path / "prepare-conda-args.txt"
    captured_script = tmp_path / "prepare-script-path.txt"

    result = _run_bash(
        _source_prepare_functions(
            f"""
            ENV_PREFIX=/tmp/fake-official-env

            run_with_ascend_env() {{
                "$@"
            }}

            conda() {{
                printf '%s\n' "$@" > {shlex.quote(str(captured_args))}
                local script_file="${{@: -1}}"
                printf '%s\n' "$script_file" > {shlex.quote(str(captured_script))}
                [[ "$script_file" != "-" ]]
                [[ -f "$script_file" ]]
                grep -Fq 'print("prepare-ok")' "$script_file"
            }}

            run_in_official_env_python '/tmp/official-a:/tmp/official-b' env SAMPLE_VAR=1 <<'PY'
print("prepare-ok")
PY

            script_file=$(cat {shlex.quote(str(captured_script))})
            [[ ! -e "$script_file" ]]
            """
        )
    )

    assert result.returncode == 0
    args = captured_args.read_text(encoding="utf-8").splitlines()
    assert args[:3] == ["run", "-p", "/tmp/fake-official-env"]
    assert args[-2] == "python"
    assert args[-1] != "-"


def test_run_in_official_runtime_python_uses_temp_script(tmp_path: Path) -> None:
    captured_args = tmp_path / "runtime-conda-args.txt"
    captured_script = tmp_path / "runtime-script-path.txt"
    captured_pythonpath = tmp_path / "runtime-pythonpath.txt"

    result = _run_bash(
        _source_run_official_functions(
            f"""
            run_in_official_runtime() {{
                local pythonpath_prefix=$1
                shift
                printf '%s\n' "$pythonpath_prefix" > {shlex.quote(str(captured_pythonpath))}
                printf '%s\n' "$@" > {shlex.quote(str(captured_args))}
                local script_file="${{@: -1}}"
                printf '%s\n' "$script_file" > {shlex.quote(str(captured_script))}
                [[ "$script_file" != "-" ]]
                [[ -f "$script_file" ]]
                grep -Fq 'print("runtime-ok")' "$script_file"
            }}

            run_in_official_runtime_python '/tmp/runtime-a:/tmp/runtime-b' env SAMPLE_VAR=1 <<'PY'
print("runtime-ok")
PY

            script_file=$(cat {shlex.quote(str(captured_script))})
            [[ ! -e "$script_file" ]]
            """
        )
    )

    assert result.returncode == 0
    assert (
        captured_pythonpath.read_text(encoding="utf-8").strip()
        == "/tmp/runtime-a:/tmp/runtime-b"
    )
    args = captured_args.read_text(encoding="utf-8").splitlines()
    assert args[:2] == ["env", "SAMPLE_VAR=1"]
    assert args[-2] == "python"
    assert args[-1] != "-"


def test_run_in_official_runtime_exports_vllm_version(tmp_path: Path) -> None:
    captured_args = tmp_path / "runtime-env-conda-args.txt"
    captured_version = tmp_path / "runtime-vllm-version.txt"

    result = _run_bash(
        _source_run_official_functions(
            f"""
            GOAL_BASELINE_ENV_PREFIX=/tmp/fake-official-env
            OFFICIAL_RUNTIME_CWD=/tmp
            OFFICIAL_VLLM_CACHE_ROOT=/tmp/fake-official-cache
            OFFICIAL_CORE_VERSION=0.11.0
            ASCEND_TOOLKIT_SET_ENV=/nonexistent
            ASCEND_ATB_SET_ENV=/nonexistent

            # run_in_official_runtime runs commands directly (not via conda).
            # Mock vllm to capture the VLLM_VERSION env var and args.
            vllm() {{
                printf '%s\n' "$VLLM_VERSION" > {shlex.quote(str(captured_version))}
                printf '%s\n' "$@" > {shlex.quote(str(captured_args))}
            }}


            run_in_official_runtime '/tmp/runtime-a:/tmp/runtime-b' vllm serve --model foo
            """
        )
    )

    assert result.returncode == 0
    assert captured_version.read_text(encoding="utf-8").strip() == "0.11.0"
    args = captured_args.read_text(encoding="utf-8").splitlines()
    assert "serve" in args
    assert "--model" in args
    assert "foo" in args


def test_run_client_command_uses_bench_cli_shape_for_serve(tmp_path: Path) -> None:
    captured_pythonpath = tmp_path / "client-pythonpath.txt"
    captured_args = tmp_path / "client-args.txt"

    result = _run_bash(
        _source_run_client_functions(
            f"""
            BENCHMARK_TYPE=serve
            OFFICIAL_RUNTIME_PYTHONPATH=/tmp/runtime-a:/tmp/runtime-b
            VLLM_CLI_COMPAT=/tmp/run_vllm_cli_compat.py
            RESULT_DIR=/tmp/result-dir
            RAW_RESULT_FILE=/tmp/result-dir/raw_benchmark_result.json
            CLIENT_ARGS='--backend vllm --model foo/bar'

            run_in_official_runtime() {{
                local pythonpath_prefix=$1
                shift
                printf '%s\n' "$pythonpath_prefix" > {shlex.quote(str(captured_pythonpath))}
                printf '%s\n' "$@" > {shlex.quote(str(captured_args))}
            }}

            run_client_command
            """
        )
    )

    assert result.returncode == 0
    assert (
        captured_pythonpath.read_text(encoding="utf-8").strip()
        == "/tmp/runtime-a:/tmp/runtime-b"
    )
    assert captured_args.read_text(encoding="utf-8").splitlines()[:5] == [
        "python",
        "/tmp/run_vllm_cli_compat.py",
        "bench",
        "serve",
        "--save-result",
    ]


def test_run_client_command_prepares_single_card_runtime_for_offline_benchmarks(
    tmp_path: Path,
) -> None:
    captured_events = tmp_path / "offline-events.txt"
    captured_args = tmp_path / "offline-client-args.txt"

    result = _run_bash(
        _source_run_client_functions(
            f"""
            BENCHMARK_TYPE=latency
            OFFICIAL_RUNTIME_PYTHONPATH=/tmp/runtime-a:/tmp/runtime-b
            VLLM_CLI_COMPAT=/tmp/run_vllm_cli_compat.py
            RAW_RESULT_FILE=/tmp/result-dir/raw_benchmark_result.json
            CLIENT_ARGS='--model foo/bar'
            RESOURCE_BUSY_EXIT_CODE=75
            DEVICE_SELECTION_RETRIES=20
            ASCEND_RUNTIME_READY_TIMEOUT_SECONDS=30
            ASCEND_VISIBLE_DEVICES=3
            ASCEND_RT_VISIBLE_DEVICES=3

            wait_for_single_card_ascend_device() {{
                printf 'select\n' >> {shlex.quote(str(captured_events))}
            }}

            wait_for_ascend_runtime_ready() {{
                printf 'ready\n' >> {shlex.quote(str(captured_events))}
            }}

            run_in_official_runtime() {{
                local pythonpath_prefix=$1
                shift
                printf 'run\n' >> {shlex.quote(str(captured_events))}
                printf '%s\n' "$@" > {shlex.quote(str(captured_args))}
            }}

            run_client_command
            """
        )
    )

    assert result.returncode == 0
    assert captured_events.read_text(encoding="utf-8").splitlines() == [
        "select",
        "ready",
        "run",
    ]
    assert captured_args.read_text(encoding="utf-8").splitlines()[:5] == [
        "python",
        "/tmp/run_vllm_cli_compat.py",
        "bench",
        "latency",
        "--output-json",
    ]


def test_run_client_command_fails_closed_on_weak_ref_failure(
    tmp_path: Path,
) -> None:
    first_args = tmp_path / "offline-first-args.txt"
    result = _run_bash(
        _source_run_client_functions(
            f"""
            BENCHMARK_TYPE=throughput
            OFFICIAL_RUNTIME_PYTHONPATH=/tmp/runtime-a:/tmp/runtime-b
            VLLM_CLI_COMPAT=/tmp/run_vllm_cli_compat.py
            RAW_RESULT_FILE=/tmp/result-dir/raw_benchmark_result.json
            CLIENT_ARGS='--model foo/bar'
            RESOURCE_BUSY_EXIT_CODE=75
            DEVICE_SELECTION_RETRIES=20
            ASCEND_RUNTIME_READY_TIMEOUT_SECONDS=30
            ASCEND_VISIBLE_DEVICES=3
            ASCEND_RT_VISIBLE_DEVICES=3

            wait_for_single_card_ascend_device() {{
                return 0
            }}

            wait_for_ascend_runtime_ready() {{
                return 0
            }}

            run_in_official_runtime() {{
                local pythonpath_prefix=$1
                shift
                printf '%s\n' "$@" > {shlex.quote(str(first_args))}
                cat <<'EOF' >&2
AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'weak_ref_tensor'
EOF
                return 1
            }}

            run_client_command
            """
        ),
        check=False,
    )

    assert result.returncode != 0
    assert "--enforce-eager" not in first_args.read_text(encoding="utf-8").splitlines()
    assert "weak_ref_tensor" in result.stderr


def test_configure_single_card_ascend_device_derives_from_generic_visible_devices() -> (
    None
):
    result = _run_bash(
        _source_run_official_functions(
            """
            unset ASCEND_RT_VISIBLE_DEVICES
            ASCEND_VISIBLE_DEVICES=' 2, 5 '
            CHIP_COUNT=2
            verify_explicit_multicard_scope_idle() { return 0; }

            configure_single_card_ascend_device

            printf 'devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
            printf 'preflight=%s\n' "$VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE"
            """
        )
    )

    assert result.stdout.splitlines()[-2:] == [
        "devices=2,5",
        "preflight=npu:0",
    ]


def test_multicard_scope_requires_explicit_exact_cardinality() -> None:
    missing = _run_bash(
        _source_run_official_functions(
            """
            unset ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES
            CHIP_COUNT=2
            configure_single_card_ascend_device
            """
        ),
        check=False,
    )
    assert missing.returncode != 0
    assert "requires an explicit" in missing.stderr

    mismatch = _run_bash(
        _source_run_official_functions(
            """
            ASCEND_RT_VISIBLE_DEVICES=0
            CHIP_COUNT=2
            configure_single_card_ascend_device
            """
        ),
        check=False,
    )
    assert mismatch.returncode != 0
    assert "does not match chip_count 2" in mismatch.stderr


def test_explicit_multicard_scope_fails_when_idle_state_cannot_be_proven() -> None:
    result = _run_bash(
        _source_run_official_functions(
            """
            ASCEND_RT_VISIBLE_DEVICES=0,1
            CHIP_COUNT=2
            resolve_npu_smi_bin() { return 1; }
            configure_single_card_ascend_device
            """
        ),
        check=False,
    )
    assert result.returncode != 0
    assert "required to prove" in result.stderr


def test_explicit_multicard_scope_checks_each_npu_process_table(tmp_path: Path) -> None:
    fake_npu_smi = tmp_path / "npu-smi"

    def write_fake(*, busy: bool) -> None:
        process_row = "| 1 0 | 4321 | python |" if busy else ""
        fake_npu_smi.write_text(
            "#!/bin/bash\n"
            'if [[ "${2:-}" == "-m" ]]; then\n'
            "  printf '0 0 0 Ascend\\n1 0 1 Ascend\\n'\n"
            "else\n"
            "  printf '| NPU | Process id | Process name |\\n'\n"
            f"  printf '%s\\n' {shlex.quote(process_row)}\n"
            "fi\n",
            encoding="utf-8",
        )
        fake_npu_smi.chmod(0o755)

    snippet = _source_run_official_functions(
        f"""
        HOST_PYTHON_BIN={shlex.quote(sys.executable)}
        NPU_SMI_TIMEOUT_SECONDS=2
        resolve_npu_smi_bin() {{ printf '%s\n' {shlex.quote(str(fake_npu_smi))}; }}
        verify_explicit_multicard_scope_idle 0,1
        """
    )
    write_fake(busy=False)
    assert _run_bash(snippet).returncode == 0

    write_fake(busy=True)
    busy = _run_bash(snippet, check=False)
    assert busy.returncode != 0
    assert "active processes: [1]" in busy.stderr


def test_configure_single_card_ascend_device_selects_detected_device() -> None:
    result = _run_bash(
        _source_run_official_functions(
            """
            unset ASCEND_RT_VISIBLE_DEVICES
            unset ASCEND_VISIBLE_DEVICES

            resolve_npu_smi_bin() {
                printf '/tmp/fake-npu-smi\n'
            }

            select_ascend_device() {
                printf '3\tidle\n'
            }

            configure_single_card_ascend_device

            printf 'devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
            printf 'preflight=%s\n' "$VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE"
            """
        )
    )

    assert result.stdout.splitlines()[-2:] == [
        "devices=3",
        "preflight=npu:0",
    ]


def test_configure_single_card_ascend_device_passes_attempt_to_selector() -> None:
    result = _run_bash(
        _source_run_official_functions(
            """
            unset ASCEND_RT_VISIBLE_DEVICES
            unset ASCEND_VISIBLE_DEVICES

            resolve_npu_smi_bin() {
                printf '/tmp/fake-npu-smi\n'
            }

            select_ascend_device() {
                printf '%s\tidle\n' "$1"
            }

            configure_single_card_ascend_device 4

            printf 'devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
            """
        )
    )

    assert result.stdout.splitlines()[-1] == "devices=4"


def test_configure_single_card_ascend_device_reuses_preferred_device_from_state_file(
    tmp_path: Path,
) -> None:
    preference_file = tmp_path / "preferred-ascend-device"
    snippet = """
            unset ASCEND_RT_VISIBLE_DEVICES
            unset ASCEND_VISIBLE_DEVICES
            GOAL_BASELINE_DEVICE_PREFERENCE_FILE=__PREFERENCE_FILE__

            resolve_npu_smi_bin() {
                printf '/tmp/fake-npu-smi\n'
            }

            select_ascend_device() {
                if [[ -n "${3:-}" ]]; then
                    printf '%s\tpreferred-idle\n' "$3"
                else
                    printf '%s\tidle\n' "$1"
                fi
            }

            configure_single_card_ascend_device 1
            printf 'first=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"

            configure_single_card_ascend_device 2
            printf 'second=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"

            printf 'stored=%s\n' "$(cat "$GOAL_BASELINE_DEVICE_PREFERENCE_FILE")"
            """.replace("__PREFERENCE_FILE__", shlex.quote(str(preference_file)))

    result = _run_bash(_source_run_official_functions(snippet))

    tracked_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("first=", "second=", "stored="))
    ]

    assert tracked_lines == [
        "first=1",
        "second=1",
        "stored=1",
    ]


def test_configure_single_card_ascend_device_returns_busy_status_when_all_devices_busy() -> (
    None
):
    result = _run_bash(
        _source_run_official_functions(
            """
            unset ASCEND_RT_VISIBLE_DEVICES
            unset ASCEND_VISIBLE_DEVICES

            resolve_npu_smi_bin() {
                printf '/tmp/fake-npu-smi\n'
            }

            select_ascend_device() {
                printf '__ALL_BUSY__\t0,1,2\n'
            }

            if configure_single_card_ascend_device 1; then
                echo 'status=unexpected-success'
            else
                echo "status=$?"
            fi

            printf 'devices=%s\n' "${ASCEND_RT_VISIBLE_DEVICES-<unset>}"
            printf 'preflight=%s\n' "${VLLM_ASCEND_TORCH_PREFLIGHT_DEVICE-<unset>}"
            printf 'reason=%s\n' "${GOAL_BASELINE_DEVICE_SELECTION_REASON-<unset>}"
            """
        )
    )

    tracked_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(("status=", "devices=", "preflight=", "reason="))
    ]

    assert tracked_lines == [
        "status=75",
        "devices=<unset>",
        "preflight=<unset>",
        "reason=all-busy",
    ]


def test_configure_single_card_ascend_device_logs_npu_smi_fallback_reason() -> None:
    result = _run_bash(
        _source_run_official_functions(
            """
            unset ASCEND_RT_VISIBLE_DEVICES
            unset ASCEND_VISIBLE_DEVICES

            resolve_npu_smi_bin() {
                printf '/tmp/fake-npu-smi\n'
            }

            select_ascend_device() {
                printf '3\tdevnode-round-robin+npu-smi-device-used\n'
            }

            configure_single_card_ascend_device

            printf 'devices=%s\n' "$ASCEND_RT_VISIBLE_DEVICES"
            """
        )
    )

    assert "devices=3" in result.stdout.splitlines()
    assert (
        "npu-smi could not inspect busy devices for the current user" in result.stderr
    )


def test_select_ascend_device_reports_all_busy_with_fake_npu_smi(
    tmp_path: Path,
) -> None:
    fake_npu_smi = tmp_path / "npu-smi"
    fake_npu_smi.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ \"${1:-}\" == \"info\" && \"${2:-}\" == \"-m\" ]]; then
cat <<'EOF'
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
0                              0                              0                              Ascend 910B3
1                              0                              1                              Ascend 910B3
EOF
elif [[ \"${1:-}\" == \"info\" ]]; then
cat <<'EOF'
+------------------------------------------------------------------------------------------------+
| npu-smi 25.3.rc1                 Version: 25.3.rc1                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 90.4        32                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          41697/ 65536         |
+===========================+===============+====================================================+
| 1     910B3               | OK            | 92.9        33                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          40844/ 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
| 0       0                 | 111           | python                   | 37974                   |
+===========================+===============+====================================================+
| 1       0                 | 222           | python                   | 37286                   |
+===========================+===============+====================================================+
EOF
else
    exit 1
fi
""",
        encoding="utf-8",
    )
    fake_npu_smi.chmod(0o755)

    result = _run_bash(
        _source_run_official_functions(
            f"""
            HOST_PYTHON_BIN={shlex.quote(sys.executable)}
            output=$(select_ascend_device 1 {shlex.quote(str(fake_npu_smi))})
            printf 'output=%s\n' "$output"
            """
        )
    )

    assert result.stdout.splitlines()[-1] == "output=__ALL_BUSY__\t0,1"


def test_select_ascend_device_prefers_previously_selected_idle_device(
    tmp_path: Path,
) -> None:
    fake_npu_smi = tmp_path / "npu-smi"
    fake_npu_smi.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ \"${1:-}\" == \"info\" && \"${2:-}\" == \"-m\" ]]; then
cat <<'EOF'
NPU ID                         Chip ID                        Chip Logic ID                  Chip Name
0                              0                              0                              Ascend 910B3
1                              0                              1                              Ascend 910B3
EOF
elif [[ \"${1:-}\" == \"info\" ]]; then
cat <<'EOF'
+------------------------------------------------------------------------------------------------+
| npu-smi 25.3.rc1                 Version: 25.3.rc1                                             |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 90.4        32                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          1024 / 65536         |
+===========================+===============+====================================================+
| 1     910B3               | OK            | 92.9        33                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          2048 / 65536         |
+===========================+===============+====================================================+
+---------------------------+---------------+----------------------------------------------------+
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
+===========================+===============+====================================================+
EOF
else
    exit 1
fi
""",
        encoding="utf-8",
    )
    fake_npu_smi.chmod(0o755)

    result = _run_bash(
        _source_run_official_functions(
            f"""
            HOST_PYTHON_BIN={shlex.quote(sys.executable)}
            output=$(select_ascend_device 1 {shlex.quote(str(fake_npu_smi))} 1)
            printf 'output=%s\n' "$output"
            """
        )
    )

    assert result.stdout.splitlines()[-1] == "output=1\tpreferred-idle"


def test_normalize_engine_version_rejects_dev_and_strips_v_prefix() -> None:
    result = _run_bash(
        _source_run_official_version_functions(
            """
            printf 'normalized=%s\n' "$(normalize_engine_version 'v0.11.0')"
            if is_valid_engine_version dev; then
                echo 'dev=valid'
            else
                echo 'dev=invalid'
            fi
            """
        )
    )

    assert result.stdout.splitlines() == [
        "normalized=0.11.0",
        "dev=invalid",
    ]


def test_wait_for_server_exits_when_server_process_is_gone(tmp_path: Path) -> None:
    stderr_file = tmp_path / "wait-for-server.stderr"

    result = _run_bash(
        _source_run_official_version_functions(
            f"""
            READY_TIMEOUT_SECONDS=30
            SERVER_PID=999999

            curl() {{
                printf 'curl-noise\n' >&2
                return 1
            }}

            if wait_for_server 127.0.0.1 8000 2>{shlex.quote(str(stderr_file))}; then
                exit 1
            fi

            grep -Fq 'Official baseline server exited before becoming ready at 127.0.0.1:8000' {shlex.quote(str(stderr_file))}
            ! grep -Fq 'curl-noise' {shlex.quote(str(stderr_file))}
            """
        )
    )

    assert result.returncode == 0


def test_wait_for_server_returns_resource_busy_status_when_log_matches(
    tmp_path: Path,
) -> None:
    stderr_file = tmp_path / "wait-for-server-resource-busy.stderr"
    server_log = tmp_path / "server.stdout.log"
    server_log.write_text(
        "RuntimeError: Initialize: error code is 507899\nResource_Busy(EL0005): The resources are busy.\n",
        encoding="utf-8",
    )

    result = _run_bash(
        _source_run_official_version_functions(
            f"""
            READY_TIMEOUT_SECONDS=30
            RESOURCE_BUSY_EXIT_CODE=75
            SERVER_PID=999999
            SERVER_STDOUT_LOG={shlex.quote(str(server_log))}

            curl() {{
                printf 'curl-noise\n' >&2
                return 1
            }}

            wait_for_server 127.0.0.1 8000 2>{shlex.quote(str(stderr_file))}
            status=$?
            printf 'status=%s\n' "$status"
            [[ "$status" == '75' ]]
            grep -Fq 'Resource_Busy(EL0005): The resources are busy.' {shlex.quote(str(stderr_file))}
            """
        ),
        check=False,
    )

    assert result.returncode == 0


def test_official_runner_preserves_failed_server_wait_status() -> None:
    script = RUN_OFFICIAL_SCRIPT.read_text(encoding="utf-8")
    serve_block = script[script.index('case "$BENCHMARK_TYPE" in') :]
    wait_block = serve_block[
        serve_block.index('if wait_for_server "$CLIENT_HOST" "$CLIENT_PORT"; then') :
    ]
    wait_block = wait_block[: wait_block.index('if [[ "$server_wait_status" -eq')]

    assert "else" in wait_block
    assert "server_wait_status=$?" in wait_block


def test_wait_for_server_fails_fast_on_fatal_startup_log(tmp_path: Path) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "EngineCore failed to start\n"
        "RuntimeError: Worker failed with error 'aclnnMoeInitRoutingCustom not in libopapi.so'\n",
        encoding="utf-8",
    )

    result = _run_bash(
        _source_run_official_version_functions(
            f"""
            READY_TIMEOUT_SECONDS=30
            READY_STATUS_INTERVAL_SECONDS=30
            RESOURCE_BUSY_EXIT_CODE=75
            SERVER_PID=$$
            SERVER_STDOUT_LOG={shlex.quote(str(server_log))}
            curl() {{ return 1; }}
            wait_for_server 127.0.0.1 8000
            """
        ),
        check=False,
    )

    assert result.returncode == 1
    assert "fatal startup error" in result.stderr


def test_missing_runtime_operator_is_not_classified_as_resource_busy(
    tmp_path: Path,
) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Engine core initialization failed\n"
        "aclnnMoeInitRoutingCustom or aclnnMoeInitRoutingCustomGetWorkspaceSize not in libopapi.so\n",
        encoding="utf-8",
    )

    result = _run_bash(
        _source_run_official_version_functions(
            f"""
            ! server_log_indicates_resource_busy {shlex.quote(str(server_log))}
            server_log_indicates_fatal_startup_error {shlex.quote(str(server_log))}
            """
        )
    )

    assert result.returncode == 0


def test_official_runner_fail_closes_and_exports_pinned_runtime_environment() -> None:
    script = RUN_OFFICIAL_SCRIPT.read_text(encoding="utf-8")

    assert "Unsupported official runtime environment" in script
    assert "OFFICIAL_RUNTIME_VLLM_BATCH_INVARIANT=1" in script
    assert (
        'export VLLM_BATCH_INVARIANT="$OFFICIAL_RUNTIME_VLLM_BATCH_INVARIANT"' in script
    )
    assert "unset VLLM_BATCH_INVARIANT" in script


def test_wait_for_ascend_runtime_ready_returns_resource_busy_status(
    tmp_path: Path,
) -> None:
    runtime_log = tmp_path / "runtime-ready.log"

    result = _run_bash(
        _source_run_official_version_functions(
            f"""
            ASCEND_RUNTIME_READY_TIMEOUT_SECONDS=1
            ASCEND_RUNTIME_READY_POLL_SECONDS=1
            RESOURCE_BUSY_EXIT_CODE=75
            RUNTIME_READY_LOG={shlex.quote(str(runtime_log))}
            OFFICIAL_RUNTIME_PYTHONPATH=/tmp/runtime-a:/tmp/runtime-b

            run_in_official_runtime_python() {{
                cat <<'EOF' >&2
RuntimeError: Initialize: error code is 507899
Resource_Busy(EL0005): The resources are busy.
EOF
                return 1
            }}

            wait_for_ascend_runtime_ready
            status=$?
            printf 'status=%s\n' "$status"
            [[ "$status" == '75' ]]
            """
        ),
        check=False,
    )

    assert result.returncode == 0


def test_normalized_server_parameters_json_preserves_graph_mode(
    tmp_path: Path,
) -> None:
    same_spec_file = tmp_path / "resolved_same_spec.json"
    same_spec_file.write_text(
        '{"resolved_server_parameters":{"model":"foo/bar","port":8000,'
        '"enforce_eager":""}}',
        encoding="utf-8",
    )

    result = _run_bash(
        _source_run_official_runtime_model_functions(
            f"""
            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            HOST_PYTHON_BIN={shlex.quote(sys.executable)}
            BENCHMARK_TYPE=serve
            SAME_SPEC_FILE={shlex.quote(str(same_spec_file))}

            server_json=$(normalized_server_parameters_json)
            printf '%s\n' "$server_json"
            grep -Fq '"enforce_eager":""' <<< "$server_json"
            """
        )
    )

    assert result.returncode == 0


def test_normalized_client_parameters_json_carries_offline_runtime_knobs(
    tmp_path: Path,
) -> None:
    same_spec_file = tmp_path / "resolved_same_spec.json"
    same_spec_file.write_text(
        '{"resolved_server_parameters":{"model":"/models/qwen",'
        '"dtype":"float16","max_model_len":32768,'
        '"tensor_parallel_size":1},'
        '"resolved_client_parameters":{"input_len":1024,'
        '"output_len":128,"batch_size":8}}',
        encoding="utf-8",
    )

    result = _run_bash(
        _source_run_official_runtime_model_functions(
            f"""
            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            HOST_PYTHON_BIN={shlex.quote(sys.executable)}
            SAME_SPEC_FILE={shlex.quote(str(same_spec_file))}
            BENCHMARK_TYPE=latency
            CLIENT_READY_CHECK_TIMEOUT_SECONDS=900
            OFFICIAL_VLLM_WORKTREE={shlex.quote(str(tmp_path / "vllm"))}
            OFFICIAL_BENCHMARK_DATASET_ROOT={shlex.quote(str(tmp_path / "datasets"))}

            client_json=$(normalized_client_parameters_json)
            printf '%s\n' "$client_json"
            grep -Fq '"dtype":"float16"' <<< "$client_json"
            grep -Fq '"max_model_len":32768' <<< "$client_json"
            grep -Fq '"tensor_parallel_size":1' <<< "$client_json"
            grep -Fq '"model":"/models/qwen"' <<< "$client_json"
            """
        )
    )

    assert result.returncode == 0


def test_official_runner_has_fail_closed_trace_replay_branch() -> None:
    source = RUN_OFFICIAL_SCRIPT.read_text(encoding="utf-8")

    assert "verify_trace_asset(get_trace_target" in source
    assert "python -m vllm_hust_benchmark.trace_replay replay" in source
    assert '--overflow-policy "$TRACE_OVERFLOW_POLICY"' in source
    assert '--summary-output "$RAW_RESULT_FILE"' in source
    assert "official trace asset not found" in source
    assert "revision=os.environ['MODEL_REVISION']" in source
    assert "model_artifact_provenance.json" in source
    assert "startup_evidence.json" in source
    assert "DECLARED_CORE_SOURCE_COMMIT" in source
    assert "DECLARED_BACKEND_SOURCE_COMMIT" in source
    assert "Official vLLM source commit mismatch" in source
    assert "Official vLLM Ascend source commit mismatch" in source
    assert "trace_replay_plan.json" in source
    assert "--concurrent-requests" in source
    assert "verify_trace_runtime_packages" in source
    assert "runtime_package_provenance.json" in source
    assert "OFFICIAL_RUNTIME_IMAGE must exactly match" in source
    assert '"runtime_image_digest": os.environ["EXPECTED_IMAGE_DIGEST"]' in source
    assert '.resolved_server_parameters.host // "127.0.0.1"' in source
    assert '.resolved_client_parameters.host // "127.0.0.1"' in source


def test_resolve_runtime_model_prefers_complete_snapshot_sibling(
    tmp_path: Path,
) -> None:
    snapshots_dir = tmp_path / "hub" / "models--foo--bar" / "snapshots"
    incomplete_snapshot = snapshots_dir / "000-incomplete"
    complete_snapshot = snapshots_dir / "111-complete"
    incomplete_snapshot.mkdir(parents=True)
    complete_snapshot.mkdir(parents=True)

    (incomplete_snapshot / "config.json").write_text("{}\n", encoding="utf-8")

    (complete_snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (complete_snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (complete_snapshot / "model-00001-of-00001.safetensors").write_text(
        "weights\n", encoding="utf-8"
    )

    result = _run_bash(
        _source_run_official_runtime_model_functions(
            f"""
            MODEL=foo/bar
            OFFICIAL_MODEL_PATH=

            run_in_official_runtime() {{
                printf '%s\n' {shlex.quote(str(incomplete_snapshot))}
            }}

            resolved=$(resolve_runtime_model)
            printf 'resolved=%s\n' "$resolved"
            [[ "$resolved" == {shlex.quote(str(complete_snapshot))} ]]
            """
        )
    )

    assert result.returncode == 0


def test_trace_model_verification_writes_pinned_provenance(tmp_path: Path) -> None:
    revision = "7dd20894a642a0aa287e9827cb1a1f7f91386b67"  # pragma: allowlist secret
    model_dir = tmp_path / "model"
    download_metadata = model_dir / ".cache/huggingface/download"
    download_metadata.mkdir(parents=True)
    files = {
        "config.json": b"{}\n",
        "tokenizer_config.json": b"{}\n",
        "tokenizer.json": b"{}\n",
        "model.safetensors": b"weights\n",
    }
    for name, contents in files.items():
        (model_dir / name).write_bytes(contents)
        digest = hashlib.sha256(contents).hexdigest()
        (download_metadata / f"{name}.metadata").write_text(
            f"{revision}\n{digest}\n", encoding="utf-8"
        )

    result_dir = tmp_path / "result"
    result = _run_bash(
        _source_run_official_runtime_model_functions(
            f"""
            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            HOST_PYTHON_BIN={shlex.quote(sys.executable)}
            RESULT_DIR={shlex.quote(str(result_dir))}
            MODEL_REVISION={revision}
            verify_runtime_model_artifact {shlex.quote(str(model_dir))}
            """
        )
    )
    provenance = json.loads(
        (result_dir / "model_artifact_provenance.json").read_text(encoding="utf-8")
    )
    assert result.returncode == 0
    assert provenance["manifest"]["revision"] == revision
    assert len(provenance["model_artifact_digest"]) == 64
    assert provenance["model_artifact_digest"] in result.stderr


def test_trace_model_verification_fails_closed_on_incomplete_download(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "weights.incomplete").write_bytes(b"")
    result = _run_bash(
        _source_run_official_runtime_model_functions(
            f"""
            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            HOST_PYTHON_BIN={shlex.quote(sys.executable)}
            RESULT_DIR={shlex.quote(str(tmp_path / "result"))}
            MODEL_REVISION=7dd20894a642a0aa287e9827cb1a1f7f91386b67
            verify_runtime_model_artifact {shlex.quote(str(model_dir))}
            """
        ),
        check=False,
    )
    assert result.returncode != 0
    assert "incomplete Hugging Face downloads" in result.stderr


def test_existing_worktree_must_match_ref_and_be_tracked_clean(tmp_path: Path) -> None:
    source_repo = tmp_path / "source"
    source_repo.mkdir()
    _run_bash(f"git -C {shlex.quote(str(source_repo))} init -q")
    _run_bash(
        f"git -C {shlex.quote(str(source_repo))} config user.email test@example.com && "
        f"git -C {shlex.quote(str(source_repo))} config user.name Test"
    )
    tracked = source_repo / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    _run_bash(
        f"git -C {shlex.quote(str(source_repo))} add tracked.txt && "
        f"git -C {shlex.quote(str(source_repo))} commit -qm one && "
        f"git -C {shlex.quote(str(source_repo))} tag pinned"
    )
    worktree = tmp_path / "worktree"
    create = _run_bash(
        _source_worktree_functions(
            f"ensure_worktree {shlex.quote(str(source_repo))} "
            f"{shlex.quote(str(worktree))} pinned"
        )
    )
    assert "verified source" in create.stdout

    original_commit = _run_bash(
        f"git -C {shlex.quote(str(worktree))} rev-parse HEAD"
    ).stdout.strip()
    tracked.write_text("two\n", encoding="utf-8")
    _run_bash(
        f"git -C {shlex.quote(str(source_repo))} add tracked.txt && "
        f"git -C {shlex.quote(str(source_repo))} commit -qm two && "
        f"git -C {shlex.quote(str(source_repo))} tag -f pinned"
    )
    mismatch = _run_bash(
        _source_worktree_functions(
            f"ensure_worktree {shlex.quote(str(source_repo))} "
            f"{shlex.quote(str(worktree))} pinned"
        ),
        check=False,
    )
    assert mismatch.returncode != 0
    assert "HEAD mismatch" in mismatch.stderr
    _run_bash(f"git -C {shlex.quote(str(source_repo))} tag -f pinned {original_commit}")

    (worktree / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    dirty = _run_bash(
        _source_worktree_functions(
            f"ensure_worktree {shlex.quote(str(source_repo))} "
            f"{shlex.quote(str(worktree))} pinned"
        ),
        check=False,
    )
    assert dirty.returncode != 0
    assert "tracked modifications" in dirty.stderr


def test_trace_startup_evidence_binds_plan_sources_model_and_results(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    provenance = result_dir / "model_artifact_provenance.json"
    provenance.write_text(
        json.dumps({"model_artifact_digest": "a" * 64}), encoding="utf-8"
    )
    (result_dir / "runtime_package_provenance.json").write_text(
        json.dumps(
            {
                "runtime_packages": {"transformers": "5.5.4"},
                "runtime_image": "quay.io/ascend/vllm-ascend@sha256:" + "b" * 64,
                "runtime_image_digest": "sha256:" + "b" * 64,
                "runtime_environment": {"VLLM_BATCH_INVARIANT": "1"},
            }
        ),
        encoding="utf-8",
    )
    raw = result_dir / "raw.json"
    detail = result_dir / "detail.jsonl"
    raw.write_text('{"ok":true}\n', encoding="utf-8")
    detail.write_text('{"request":1}\n', encoding="utf-8")
    expected_raw_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    expected_detail_hash = hashlib.sha256(detail.read_bytes()).hexdigest()

    result = _run_bash(
        _source_run_official_runtime_model_functions(
            f"""
            HOST_PYTHON_BIN={shlex.quote(sys.executable)}
            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            RESULT_DIR={shlex.quote(str(result_dir))}
            RUN_ID=run-test
            TRACE_TARGET_ID=trace-test
            TRACE_ASSET_PATH=/tmp/trace
            RUNTIME_MODEL=/tmp/model
            TRACE_MAX_MODEL_LEN=1024
            TRACE_MAX_REQUESTS=2
            TRACE_MAX_CONCURRENCY=1
            TRACE_OVERFLOW_POLICY=reject
            TRACE_TIME_SCALE=1
            TRACE_MAX_INTERARRIVAL_S=1
            OFFICIAL_RUNTIME_PYTHONPATH=/tmp/runtime
            OFFICIAL_CORE_SOURCE_COMMIT={"1" * 40}
            OFFICIAL_BACKEND_SOURCE_COMMIT={"2" * 40}
            RAW_RESULT_FILE={shlex.quote(str(raw))}
            TRACE_DETAIL_RESULT_FILE={shlex.quote(str(detail))}
            run_in_official_runtime() {{
              printf '%s\n' '{{"cohort_setting_signature":"cohort-1","cohort":{{"setting_signature_payload":{{"trace_asset_sha256":"asset-1"}}}}}}'
            }}
            prepare_trace_startup_evidence
            finalize_trace_startup_evidence
            """
        )
    )
    evidence = json.loads(
        (result_dir / "startup_evidence.json").read_text(encoding="utf-8")
    )
    assert result.returncode == 0
    assert evidence["run_id"] == "run-test"
    assert evidence["engine_source_commit"] == "1" * 40
    assert evidence["plugin_source_commit"] == "2" * 40
    assert evidence["model_artifact_digest"] == "a" * 64
    assert evidence["runtime_packages"] == {"transformers": "5.5.4"}
    assert evidence["runtime_image_digest"] == "sha256:" + "b" * 64
    assert evidence["runtime_environment"] == {"VLLM_BATCH_INVARIANT": "1"}
    assert evidence["trace_asset_sha256"] == "asset-1"
    assert evidence["cohort_setting_signature"] == "cohort-1"
    assert evidence["finished_at"]
    assert evidence["result_hashes"] == {
        "raw_sha256": expected_raw_hash,
        "detail_sha256": expected_detail_hash,
    }


def test_trace_runtime_provenance_rejects_unpinned_image(tmp_path: Path) -> None:
    digest = "sha256:" + "b" * 64
    expected_image = "quay.io/ascend/vllm-ascend@" + digest
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "baseline_target": {
                    "runtime_packages": {"transformers": "5.5.4"},
                    "runtime_image": expected_image,
                    "runtime_image_digest": digest,
                }
            }
        ),
        encoding="utf-8",
    )
    result = _run_bash(
        _source_run_official_runtime_model_functions(
            f"""
            SPEC_FILE={shlex.quote(str(spec))}
            RESULT_DIR={shlex.quote(str(tmp_path / "result"))}
            OFFICIAL_RUNTIME_IMAGE=quay.io/ascend/vllm-ascend@sha256:{"c" * 64}
            verify_trace_runtime_packages
            """
        ),
        check=False,
    )
    assert result.returncode != 0
    assert "must exactly match" in result.stderr


def test_ensure_vllm_ascend_plugin_metadata_writes_entry_points(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "vllm-ascend-worktree"
    worktree_dir.mkdir()
    (worktree_dir / "vllm_ascend").mkdir()
    (worktree_dir / "setup.py").write_text(
        "entry_points={\n"
        '    "vllm.platform_plugins": [\n'
        '        "ascend = vllm_ascend:register",\n'
        "    ],\n"
        '    "vllm.general_plugins": [\n'
        '        "ascend_enhanced_model = vllm_ascend:register_model",\n'
        '        "ascend_kv_connector = vllm_ascend:register_connector",\n'
        "    ],\n"
        "}\n",
        encoding="utf-8",
    )

    result = _run_bash(
        _source_prepare_functions(
            f"""
            OFFICIAL_VLLM_ASCEND_WORKTREE={shlex.quote(str(worktree_dir))}
            OFFICIAL_VLLM_ASCEND_REF=v0.11.0
            OFFICIAL_SOC_VERSION=ascend910b3
            OFFICIAL_SLEEP_MODE_ENABLED=0
            ensure_vllm_ascend_plugin_metadata
            dist_info_dir=$(printf '%s\n' {shlex.quote(str(worktree_dir))}/vllm_ascend-0.11.0.dist-info)
            [[ -d "$dist_info_dir" ]]
            grep -Fq 'Name: vllm-ascend' "$dist_info_dir/METADATA"
            grep -Fq 'Version: 0.11.0' "$dist_info_dir/METADATA"
            grep -Fq 'ascend = vllm_ascend:register' "$dist_info_dir/entry_points.txt"
            grep -Fq 'ascend_enhanced_model = vllm_ascend:register_model' "$dist_info_dir/entry_points.txt"
            grep -Fq 'ascend_kv_connector = vllm_ascend:register_connector' "$dist_info_dir/entry_points.txt"
            grep -Fq 'vllm_ascend' "$dist_info_dir/top_level.txt"
            grep -Fq "__soc_version__ = 'ascend910b3'" {shlex.quote(str(worktree_dir))}/vllm_ascend/_build_info.py
            grep -Fq '__sleep_mode_enabled__ = False' {shlex.quote(str(worktree_dir))}/vllm_ascend/_build_info.py
            """
        )
    )

    assert result.returncode == 0
