"""Run the original smolagents Python tool only in an explicit isolated Docker.

Trusted operator calls this helper. Model-generated code is stdin to the guest,
never a host command. The mount contains only the pinned framework environment,
not original GAIA metadata, references, credentials or the Docker socket.
"""

import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import uuid


GUEST = """
import json,sys
from smolagents import PythonInterpreterTool
code=sys.stdin.read()
print(json.dumps({'tool_output':PythonInterpreterTool().forward(code)}))
"""


def command(*, docker_host, image, framework_venv, name):
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", image):
        raise ValueError("explicit immutable local image digest required")
    if not docker_host.startswith("unix:///") or "\n" in docker_host:
        raise ValueError("explicit local Unix Docker socket required")
    venv = PurePosixPath(framework_venv)
    if not venv.is_absolute() or any(c in str(venv) for c in [":", ",", "\n"]):
        raise ValueError("absolute dedicated framework-only environment required")
    return [
        "docker",
        "-H",
        docker_host,
        "run",
        "--name",
        name,
        "-i",
        "--pull",
        "never",
        "--log-driver",
        "none",
        "--network",
        "none",
        "--read-only",
        "--user",
        "65534:65534",
        "--cpus",
        "1",
        "--memory",
        "256m",
        "--pids-limit",
        "64",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=64m",
        "-e",
        "HOME=/tmp",
        "-e",
        "HF_HOME=/tmp/hf",
        "-e",
        "HF_TOKEN=",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-v",
        str(venv) + ":/work/venv:ro",
        "--entrypoint",
        "/work/venv/bin/python",
        image,
        "-IB",
        "-c",
        GUEST,
    ]


def run_code(code, *, docker_host, image, framework_venv, output, timeout=60):
    if not isinstance(code, str) or not code or len(code.encode()) > 65536:
        raise ValueError("nonempty Python code limited to64KiB required")
    if type(timeout) is not int or not 1 <= timeout <= 300:
        raise ValueError("explicit bounded tool timeout required")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh private output directory required")
    if output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("private tool output must remain outside repository")
    name = "gaia-python-" + uuid.uuid4().hex
    argv = command(
        docker_host=docker_host, image=image, framework_venv=framework_venv, name=name
    )
    output.mkdir(mode=0o700, parents=True)
    (output / "code-private.py").write_text(code, encoding="utf-8")
    receipt = {"container": name, "native_gaia_task_completed": False}
    try:
        with (
            (output / "stdout-private.txt").open("xb") as stdout,
            (output / "stderr-private.txt").open("xb") as stderr,
        ):
            result = subprocess.run(
                ["prlimit", "--fsize=1048576:1048576", "--", *argv],
                input=code.encode(),
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        receipt["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError("isolated original Python tool failed; retain logs")
        raw = (output / "stdout-private.txt").read_text(encoding="utf-8")
        value = json.loads(raw)
        if set(value) != {"tool_output"} or not isinstance(value["tool_output"], str):
            raise ValueError("original tool result contract differs")
        return value["tool_output"]
    except subprocess.TimeoutExpired:
        receipt["timed_out"] = True
        raise
    finally:
        # Stop only this invocation, never the daemon or another task's container.
        # Also stop a guest left running when the bounded Docker client fails.
        receipt["cleanup_status"] = "ATTEMPTED_NOT_CONFIRMED"
        try:
            stopped = subprocess.run(
                ["docker", "-H", docker_host, "kill", name],
                capture_output=True,
                timeout=20,
                check=False,
            )
            receipt["owned_container_kill_returncode"] = stopped.returncode
            receipt["cleanup_status"] = "COMMAND_RETURNED_INSPECT_CONTAINER_IF_NONZERO"
        finally:
            (output / "receipt-private.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8"
            )
