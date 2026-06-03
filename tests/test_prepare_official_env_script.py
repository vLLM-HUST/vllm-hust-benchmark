import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "scripts/prepare-official-ascend-baseline-env.sh"
RUN_OFFICIAL_SCRIPT = REPO_ROOT / "scripts/run-official-ascend-goal-baseline.sh"


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


def _source_run_official_arg_pipeline_functions(snippet: str) -> str:
    script_path = shlex.quote(str(RUN_OFFICIAL_SCRIPT))
    return (
        r"source <(awk 'BEGIN{capture=0} /^json2args\(\) \{/ {capture=1} /^kill_server\(\) \{/ {exit} capture {print}' "
        f"{script_path}) && {snippet}"
    )


def _run_bash(command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
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


def _spawn_process_tree(tmp_path: Path) -> tuple[subprocess.Popen[str], int]:
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
        [
            "bash",
            "-lc",
            f'exec -a "vllm.entrypoints.cli.main bench serve" {shlex.quote(str(wrapper_script))}',
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    child_pid = _wait_for_pid_file(child_pid_file)
    return process, child_pid


def test_collect_process_tree_pids_includes_child_process(tmp_path: Path) -> None:
    process, child_pid = _spawn_process_tree(tmp_path)

    try:
        result = _run_bash(
            _source_prepare_functions(f"collect_process_tree_pids {process.pid}")
        )
        collected_pids = {
            int(line)
            for line in result.stdout.splitlines()
            if line.strip()
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
                            [[ "$1" == '501' || "$1" == '601' ]]
                        }

                        is_benchmark_process() {
                            [[ "$1" == '601' || "$1" == '602' ]]
                        }

                        is_managed_runner_wrapper_process() {
                            [[ "$1" == '501' || "$1" == '502' ]]
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
                "601",
                "out-of-scope:",
                "602",
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
    assert captured_pythonpath.read_text(encoding="utf-8").strip() == "/tmp/runtime-a:/tmp/runtime-b"
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

            conda() {{
                printf '%s\n' "$VLLM_VERSION" > {shlex.quote(str(captured_version))}
                printf '%s\n' "$@" > {shlex.quote(str(captured_args))}
            }}

            run_in_official_runtime '/tmp/runtime-a:/tmp/runtime-b' env SAMPLE_VAR=1 true
            """
        )
    )

    assert result.returncode == 0
    assert captured_version.read_text(encoding="utf-8").strip() == "0.11.0"
    args = captured_args.read_text(encoding="utf-8").splitlines()
    assert args[:3] == ["run", "-p", "/tmp/fake-official-env"]


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
    assert captured_pythonpath.read_text(encoding="utf-8").strip() == "/tmp/runtime-a:/tmp/runtime-b"
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


def test_run_client_command_retries_with_enforce_eager_on_weak_ref_failure(
    tmp_path: Path,
) -> None:
    first_args = tmp_path / "offline-first-args.txt"
    retry_args = tmp_path / "offline-retry-args.txt"
    first_attempt = tmp_path / "offline-first-attempt.txt"

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
                if [[ ! -f {shlex.quote(str(first_attempt))} ]]; then
                    printf '%s\n' "$@" > {shlex.quote(str(first_args))}
                    printf '1\n' > {shlex.quote(str(first_attempt))}
                    cat <<'EOF' >&2
AttributeError: '_OpNamespace' '_C_ascend' object has no attribute 'weak_ref_tensor'
EOF
                    return 1
                fi

                printf '%s\n' "$@" > {shlex.quote(str(retry_args))}
            }}

            run_client_command
            """
        )
    )

    assert result.returncode == 0
    assert "--enforce-eager" not in first_args.read_text(encoding="utf-8").splitlines()
    assert "--enforce-eager" in retry_args.read_text(encoding="utf-8").splitlines()
    assert "retrying with --enforce-eager" in result.stderr


def test_configure_single_card_ascend_device_derives_from_generic_visible_devices() -> None:
    result = _run_bash(
        _source_run_official_functions(
            """
            unset ASCEND_RT_VISIBLE_DEVICES
            ASCEND_VISIBLE_DEVICES=' 2, 5 '

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

    result = _run_bash(
        _source_run_official_functions(snippet)
    )

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


def test_configure_single_card_ascend_device_returns_busy_status_when_all_devices_busy() -> None:
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
        "npu-smi could not inspect busy devices for the current user"
        in result.stderr
    )


def test_select_ascend_device_reports_all_busy_with_fake_npu_smi(tmp_path: Path) -> None:
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
            HOST_PYTHON_BIN=$(command -v python3)
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
            HOST_PYTHON_BIN=$(command -v python3)
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


def test_wait_for_server_returns_resource_busy_status_when_log_matches(tmp_path: Path) -> None:
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


def test_wait_for_ascend_runtime_ready_returns_resource_busy_status(tmp_path: Path) -> None:
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


def test_should_force_eager_for_offline_benchmark_when_aclgraph_weak_ref_is_missing() -> None:
    result = _run_bash(
        _source_run_official_version_functions(
            """
            BENCHMARK_TYPE=throughput

            official_runtime_supports_aclgraph_weak_ref_tensor() {
                return 1
            }

            should_force_eager_for_offline_benchmark
            status=$?
            printf 'status=%s\n' "$status"
            [[ "$status" == '0' ]]
            """
        ),
        check=False,
    )

    assert result.returncode == 0
    assert "forcing --enforce-eager" in result.stderr


def test_official_runtime_supports_aclgraph_weak_ref_tensor_preserves_probe_status() -> None:
    result = _run_bash(
        _source_run_official_version_functions(
            """
            run_in_official_runtime_python() {
                return 3
            }

            if official_runtime_supports_aclgraph_weak_ref_tensor; then
                echo 'status=unexpected-success'
            else
                echo "status=$?"
            fi
            """
        ),
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == ["status=1"]


def test_should_force_eager_for_server_benchmark_when_aclgraph_weak_ref_is_missing() -> None:
    result = _run_bash(
        _source_run_official_version_functions(
            """
            BENCHMARK_TYPE=serve

            official_runtime_supports_aclgraph_weak_ref_tensor() {
                return 1
            }

            should_force_eager_for_server_benchmark
            status=$?
            printf 'status=%s\n' "$status"
            [[ "$status" == '0' ]]
            """
        ),
        check=False,
    )

    assert result.returncode == 0
    assert "forcing --enforce-eager" in result.stderr


def test_normalized_server_parameters_json_forces_eager_for_serve_when_requested(
    tmp_path: Path,
) -> None:
    same_spec_file = tmp_path / "resolved_same_spec.json"
    same_spec_file.write_text(
        '{"resolved_server_parameters":{"model":"foo/bar","port":8000}}',
        encoding="utf-8",
    )

    result = _run_bash(
        _source_run_official_runtime_model_functions(
            f"""
            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            HOST_PYTHON_BIN=$(command -v python3)
            BENCHMARK_TYPE=serve
            SAME_SPEC_FILE={shlex.quote(str(same_spec_file))}

            official_runtime_supports_aclgraph_weak_ref_tensor() {{
                return 1
            }}

            server_json=$(normalized_server_parameters_json)
            printf '%s\n' "$server_json"
            python3 - <<'PY' "$server_json"
import json
import sys

payload = json.loads(sys.argv[1])
assert payload["enforce_eager"] == ""
assert payload["model"] == "foo/bar"
assert payload["port"] == 8000
PY
            """
        )
    )

    assert result.returncode == 0
    assert "forcing --enforce-eager" in result.stderr


def test_server_arg_pipeline_preserves_json_output_when_force_eager_is_requested(
    tmp_path: Path,
) -> None:
    same_spec_file = tmp_path / "resolved_same_spec.json"
    same_spec_file.write_text(
        '{"resolved_server_parameters":{"model":"foo/bar","port":8000,"host":"0.0.0.0"}}',
        encoding="utf-8",
    )

    result = _run_bash(
        _source_run_official_arg_pipeline_functions(
            f"""
            REPO_ROOT={shlex.quote(str(REPO_ROOT))}
            HOST_PYTHON_BIN=$(command -v python3)
            BENCHMARK_TYPE=serve
            SAME_SPEC_FILE={shlex.quote(str(same_spec_file))}
            CLIENT_READY_CHECK_TIMEOUT_SECONDS=900
            OFFICIAL_VLLM_WORKTREE=/tmp/vllm-test-worktree
            OFFICIAL_BENCHMARK_DATASET_ROOT=/tmp/vllm-test-datasets

            official_runtime_supports_aclgraph_weak_ref_tensor() {{
                return 1
            }}

            server_args=$(json2args "$(normalized_server_parameters_json | jq -c 'del(.disable_log_requests)')")
            printf '%s\n' "$server_args"
            grep -Fq -- '--model foo/bar' <<< "$server_args"
            grep -Fq -- '--host 0.0.0.0' <<< "$server_args"
            grep -Fq -- '--port 8000' <<< "$server_args"
            grep -Fq -- '--enforce-eager' <<< "$server_args"
            """
        )
    )

    assert result.returncode == 0
    assert "forcing --enforce-eager" in result.stderr


def test_resolve_runtime_model_prefers_complete_snapshot_sibling(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "hub" / "models--foo--bar" / "snapshots"
    incomplete_snapshot = snapshots_dir / "000-incomplete"
    complete_snapshot = snapshots_dir / "111-complete"
    incomplete_snapshot.mkdir(parents=True)
    complete_snapshot.mkdir(parents=True)

    (incomplete_snapshot / "config.json").write_text("{}\n", encoding="utf-8")

    (complete_snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (complete_snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (complete_snapshot / "model-00001-of-00001.safetensors").write_text("weights\n", encoding="utf-8")

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


def test_ensure_vllm_ascend_plugin_metadata_writes_entry_points(tmp_path: Path) -> None:
    worktree_dir = tmp_path / "vllm-ascend-worktree"
    worktree_dir.mkdir()
    (worktree_dir / "vllm_ascend").mkdir()
    (worktree_dir / "setup.py").write_text(
        "entry_points={\n"
        "    \"vllm.platform_plugins\": [\n"
        "        \"ascend = vllm_ascend:register\",\n"
        "    ],\n"
        "    \"vllm.general_plugins\": [\n"
        "        \"ascend_enhanced_model = vllm_ascend:register_model\",\n"
        "        \"ascend_kv_connector = vllm_ascend:register_connector\",\n"
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
            grep -Fq "__device_type__ = 'A2'" {shlex.quote(str(worktree_dir))}/vllm_ascend/_build_info.py
            grep -Fq "__soc_version__ = 'ascend910b3'" {shlex.quote(str(worktree_dir))}/vllm_ascend/_build_info.py
            grep -Fq '__sleep_mode_enabled__ = False' {shlex.quote(str(worktree_dir))}/vllm_ascend/_build_info.py
            """
        )
    )

    assert result.returncode == 0


def test_run_official_script_uses_configured_official_refs_for_worktrees() -> None:
    script_text = RUN_OFFICIAL_SCRIPT.read_text(encoding="utf-8")

    assert (
        'ensure_worktree "$OFFICIAL_VLLM_REPO" "$OFFICIAL_VLLM_WORKTREE" "$OFFICIAL_VLLM_REF"'
        in script_text
    )
    assert (
        'ensure_worktree "$OFFICIAL_VLLM_ASCEND_REPO" "$OFFICIAL_VLLM_ASCEND_WORKTREE" "$OFFICIAL_VLLM_ASCEND_REF"'
        in script_text
    )
    assert 'OFFICIAL_VLLM_REF="$OFFICIAL_VLLM_REF" \\' in script_text
    assert 'OFFICIAL_VLLM_ASCEND_REF="$OFFICIAL_VLLM_ASCEND_REF" \\' in script_text
    assert 'OFFICIAL_VLLM_WORKTREE="$OFFICIAL_VLLM_WORKTREE" \\' in script_text
    assert 'OFFICIAL_VLLM_ASCEND_WORKTREE="$OFFICIAL_VLLM_ASCEND_WORKTREE" \\' in script_text


def test_prepare_script_defaults_match_current_official_baseline_dependency_versions() -> None:
    result = _run_bash(
        _source_prepare_functions(
            """
            printf '%s\n' \
              "$OFFICIAL_TRANSFORMERS_VERSION" \
              "$OFFICIAL_COMPRESSED_TENSORS_VERSION" \
              "$OFFICIAL_DEPYF_VERSION" \
              "$OFFICIAL_LLGUIDANCE_VERSION" \
              "$OFFICIAL_XGRAMMAR_VERSION" \
              "$OFFICIAL_FASTAPI_VERSION" \
                            "$OFFICIAL_UVLOOP_TARGET" \
              "$OFFICIAL_NUMBA_VERSION" \
              "$OFFICIAL_OPENCV_VERSION"
            """
        )
    )

    assert result.stdout.splitlines() == [
        "4.57.4",
        "0.13.0",
        "0.20.0",
        "1.3.0",
        "0.1.32",
        "0.123.10",
        "uvloop",
        "0.61.2",
        "4.11.0.86",
    ]


def test_prepare_script_filters_explicitly_managed_dependency_constraints(tmp_path: Path) -> None:
    source_file = tmp_path / "requirements.in"
    target_file = tmp_path / "requirements.out"
    source_file.write_text(
        "\n".join(
            [
                "torch==2.9.0",
                "transformers>=4.56.0,<5",
                "compressed-tensors==0.13.0",
                "compressed_tensors>=0.11.0",
                "depyf==0.20.0",
                "llguidance>=1.3.0,<1.4.0",
                "xgrammar>=0.1.30",
                "fastapi[standard]>=0.115.0",
                "numba",
                "opencv-python-headless>=4.13.0",
                "cachetools",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = _run_bash(
        _source_prepare_functions(
            f"""
            create_filtered_requirements_file {shlex.quote(str(source_file))} {shlex.quote(str(target_file))}
            cat {shlex.quote(str(target_file))}
            """
        )
    )

    assert result.stdout.splitlines() == ["cachetools"]


def test_prepare_script_health_check_allows_local_version_suffix_for_base_match() -> None:
    script_text = PREPARE_SCRIPT.read_text(encoding="utf-8")

    assert "def versions_match(actual: str, expected: str) -> bool:" in script_text
    assert "actual_version.base_version == expected_version.base_version" in script_text
    assert "expected_version.local is None" in script_text


def test_prepare_script_health_check_allows_extra_general_plugins() -> None:
    script_text = PREPARE_SCRIPT.read_text(encoding="utf-8")

    assert "missing_general_plugins = sorted(set(expected_general_plugins) - set(general_plugins))" in script_text
    assert "if missing_general_plugins:" in script_text
    assert "missing required" in script_text


def test_prepare_script_health_check_requires_uvloop_runtime_dependency() -> None:
    script_text = PREPARE_SCRIPT.read_text(encoding="utf-8")

    assert "OFFICIAL_UVLOOP_TARGET=${OFFICIAL_UVLOOP_TARGET:-\"uvloop\"}" in script_text
    assert 'import uvloop' in script_text
    assert script_text.count('"$OFFICIAL_UVLOOP_TARGET"') >= 2


def test_prepare_script_health_check_requires_ascend_device_type_metadata() -> None:
    script_text = PREPARE_SCRIPT.read_text(encoding="utf-8")

    assert 'OFFICIAL_EXPECTED_ASCEND_DEVICE_TYPE="$(resolve_ascend_device_type "$OFFICIAL_SOC_VERSION")"' in script_text
    assert 'from vllm_ascend import _build_info' in script_text
    assert 'actual_device_type = getattr(_build_info, "__device_type__", None)' in script_text
    assert 'expected_device_type = os.environ["OFFICIAL_EXPECTED_ASCEND_DEVICE_TYPE"]' in script_text