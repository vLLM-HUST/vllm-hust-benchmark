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
        r"source <(awk 'BEGIN{capture=0} /^run_in_official_runtime\(\) \{/ {capture=1} /^run_server_command\(\) \{/ {exit} capture {print}' "
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
            grep -Fq "__soc_version__ = 'ascend910b3'" {shlex.quote(str(worktree_dir))}/vllm_ascend/_build_info.py
            grep -Fq '__sleep_mode_enabled__ = False' {shlex.quote(str(worktree_dir))}/vllm_ascend/_build_info.py
            """
        )
    )

    assert result.returncode == 0