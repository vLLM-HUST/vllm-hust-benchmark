"""Run the original AIDE flow for fixed NOMAD in an isolated ARM64 profile."""

import argparse
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading

from appworld_model_transport import model_proxy
from mlebench_nomad_validation import UnixHTTPConnection, frontend


START_SHA = "38b106593621f46a5a990d19ce05dc205de13743f02c81ffe7b38ba33dc4e66d"
CONFIG_SHA = "b40e0e1dd07bfbfb8140507f887bb99230d542a41030632ff757d498559e0e55"
NOTES_SHA = "0dbd723720145e77d671bbb8b0c9ba684aa565f99634b6f50b47a50b72915595"
INSTRUCTIONS_SHA = "7f6bfde87ed88d53afef1688dd9206753982baf4a4038e87d6069ab57224ec7a"
TIME = 86400
STEPS = 500
OUTPUTS = ("agent", "submission", "logs", "code")


def author_files(root):
    base = root / "agents/aide"
    for name, digest in (
        ("start.sh", START_SHA),
        ("config.yaml", CONFIG_SHA),
        ("additional_notes.txt", NOTES_SHA),
    ):
        if hashlib.sha256((base / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"original AIDE {name} does not match its pinned source")
    source = (base / "start.sh").read_text(encoding="utf-8")
    old = 'eval "$(conda shell.bash hook)" # make conda available to the shell\nconda activate agent'
    if source.count(old) != 1:
        raise ValueError("unexpected original activation block")
    source = source.replace(
        old, "# Use the isolated ARM64 Python runtime selected by the launcher."
    )
    start = source.index("# check that we can use the GPU in PyTorch")
    end = source.index("# convert $TIME_LIMIT_SECS", start)
    source = (
        source[:start]
        + "# CUDA library probes omitted in the CPU profile.\n\n"
        + source[end:]
    )
    old = 'if [ $? -eq 124 ]; then\n  echo "Timed out after $TIME_LIMIT"\nfi'
    if source.count(old) != 1:
        raise ValueError("unexpected original terminal block")
    source = source.replace(
        old,
        'status=$?\nif [ "$status" -eq 124 ]; then\n  echo "Timed out after $TIME_LIMIT"\nfi\nexit "$status"',
    )
    return source, (base / "additional_notes.txt").read_bytes()


def author_args(model, *, convert_system_to_user=False):
    if not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,150}", model) or model.startswith("-"):
        raise ValueError("exact served model ID required")
    args = [
        "agent.search.max_debug_depth=20",
        "agent.search.debug_prob=1",
        f"agent.time_limit={TIME}",
        "exec.timeout=32400",
        "copy_data=False",
        f"agent.code.model={model}",
        f"agent.feedback.model={model}",
        f"agent.steps={STEPS}",
    ]
    if convert_system_to_user:
        args.append("agent.convert_system_to_user=True")
    return args


def check_isolation():
    public = Path("/home/data")
    if (
        not os.statvfs(public).f_flag & os.ST_RDONLY
        or {name for _, name in socket.if_nameindex()} != {"lo"}
        or Path("/private").exists()
        or Path("/var/run/docker.sock").exists()
        or list(Path("/dev").glob("davinci*"))
    ):
        raise ValueError(
            "Agent needs isolated read-only public input without devices or private labels"
        )
    if (
        not (public / "description.md").is_file()
        or not Path("/home/instructions.txt").is_file()
    ):
        raise ValueError("original instructions and prepared public input are required")
    if (
        hashlib.sha256(Path("/home/instructions.txt").read_bytes()).hexdigest()
        != INSTRUCTIONS_SHA
    ):
        raise ValueError("original MLE-bench instructions do not match the source pin")
    if not sys.flags.no_site or not shutil.which("envsubst"):
        raise ValueError("start with python -S and provide GNU envsubst")


def write_shim(path, command):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f'#!/bin/sh\nexec {shlex.join(command)} "$@"\n')
    path.chmod(0o700)


def execute(args, start, notes, kwargs):
    check_isolation()
    queue = args.queue.resolve(strict=True)
    validator_socket = args.validator_socket.resolve(strict=True)
    if not validator_socket.is_socket():
        raise ValueError("isolated validator socket required")
    health = UnixHTTPConnection(validator_socket)
    try:
        health.request("GET", "/health")
        response = health.getresponse()
        if response.status != 200 or json.loads(response.read(1024)) != {
            "status": "running"
        }:
            raise ValueError("isolated submission validator not healthy")
    finally:
        health.close()
    root = args.output.resolve(strict=True)
    if (
        not root.is_dir()
        or any(root.iterdir())
        or not queue.is_dir()
        or any(queue.iterdir())
    ):
        raise ValueError("new dedicated empty output and queue directories required")
    if root == queue or root.is_relative_to(queue) or queue.is_relative_to(root):
        raise ValueError("output and queue must be separate mounts")
    for name in OUTPUTS:
        (root / name).mkdir(mode=0o700)
    agent = root / "agent"
    (agent / "bin").mkdir(mode=0o700)
    (agent / "start.sh").write_text(start, encoding="utf-8", newline="\n")
    (agent / "additional_notes.txt").write_bytes(notes)
    python = [sys.executable, "-S"]
    for binary in ("python", "python3"):
        write_shim(agent / "bin" / binary, python)
    env = dict(os.environ)
    env.update(
        AGENT_DIR=str(agent),
        CODE_DIR=str(root / "code"),
        LOGS_DIR=str(root / "logs"),
        SUBMISSION_DIR=str(root / "submission"),
        TIME_LIMIT_SECS=str(TIME),
        STEP_LIMIT=str(STEPS),
        OBFUSCATE="false",
        PATH=str(agent / "bin") + os.pathsep + env["PATH"],
    )
    with ThreadingHTTPServer(
        ("127.0.0.1", 5000), frontend(validator_socket)
    ) as validation:
        worker = threading.Thread(target=validation.serve_forever, daemon=True)
        worker.start()
        try:
            with model_proxy(
                queue,
                port=0,
                expected_model=args.model,
                timeout=args.request_timeout + 15,
            ) as port:
                bind_and_run = (
                    "from aide import backend; from aide.run import run; import sys; "
                    "from mlebench_aide_local_model import bind_local_openai; "
                    "bind_local_openai(backend, model=sys.argv.pop(1), api_base=sys.argv.pop(1), "
                    "timeout=int(sys.argv.pop(1))); run()"
                )
                write_shim(
                    agent / "bin/aide",
                    [
                        *python,
                        "-c",
                        bind_and_run,
                        args.model,
                        f"http://127.0.0.1:{port}/v1",
                        str(args.request_timeout + 20),
                    ],
                )
                return subprocess.run(
                    ["/bin/bash", str(agent / "start.sh"), *kwargs], env=env
                ).returncode
        finally:
            validation.shutdown()
            worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--validator-socket", type=Path)
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="match the trusted relay's request wait",
    )
    parser.add_argument(
        "--convert-system-to-user",
        action="store_true",
        help="Use the author's role mapping for model templates that require a user turn",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.request_timeout <= 7200:
        parser.error("--request-timeout must be between 1 and 7200 seconds")
    kwargs = author_args(args.model, convert_system_to_user=args.convert_system_to_user)
    start, notes = author_files(args.author_root.resolve(strict=True))
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "PLAN_ONLY",
                    "time_limit_seconds": TIME,
                    "step_limit": STEPS,
                    "transport_request_wait_seconds": args.request_timeout,
                    "author_aide_args": kwargs,
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                }
            )
        )
        return 0
    if None in (args.output, args.queue, args.validator_socket):
        parser.error("--execute needs --output, --queue and --validator-socket")
    code = execute(args, start, notes, kwargs)
    print(
        json.dumps(
            {"original_launcher_exit_code": code, "source_checkbox_eligible": False}
        )
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
