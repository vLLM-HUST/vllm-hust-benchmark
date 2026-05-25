import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = REPO_ROOT / "scripts/prepare-official-ascend-baseline-env.sh"


def _source_prepare_functions(snippet: str) -> str:
    script_path = shlex.quote(str(PREPARE_SCRIPT))
    return (
        "source <(awk '/^if ! command -v conda / {exit} {print}' "
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