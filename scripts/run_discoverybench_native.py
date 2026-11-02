"""Run prepared original DiscoveryBench cases with an owned Docker daemon.

Default is a plan. --preflight constructs original actors offline without model
calls. --execute runs actors, isolated original Python tools and original scoring
against an explicitly owned loopback model endpoint. No generated code runs here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from urllib.parse import urlsplit
import uuid

from appworld_model_transport import process_one, save_new
from run_discoverybench_original_actor import verify_author
from run_discoverybench_original_judge import verify_scorer


RUNTIME = (
    "run_discoverybench_original_actor.py",
    "run_discoverybench_original_judge.py",
    "discoverybench_isolated_python.py",
    "discoverybench_tool_transport.py",
    "appworld_model_transport.py",
)


def checked_file(root, item):
    path = (root / item["path"]).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("prepared file escaped its root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("prepared original file identity differs")
    return path


def load_cases(manifest):
    manifest = manifest.resolve(strict=True)
    root = manifest.parent
    value = json.loads(manifest.read_bytes())
    if (
        value.get("schema_version") != "discoverybench-original-cases/v1"
        or value.get("source_id") != "DISCOVERYBENCH"
        or value.get("original_actor_iterations") != 5
        or value.get("original_python_call_limit") != 1
        or not value.get("cases")
    ):
        raise ValueError("prepared original protocol manifest required")
    positions = [case["original_position"] for case in value["cases"]]
    if len(positions) != len(set(positions)):
        raise ValueError("duplicate original positions")
    for case in value["cases"]:
        for item in [case["metadata"], case["reference"], *case["data_files"]]:
            checked_file(root, item)
        if not case["data_files"]:
            raise ValueError("original scientific data required")
    return root, value["cases"]


def daemon_path(path, host_root, daemon_root):
    path = path.resolve(strict=True)
    if not path.is_relative_to(host_root):
        raise ValueError("mount must be within the explicitly mapped host root")
    mapped = daemon_root / path.relative_to(host_root)
    if any(char in str(mapped) for char in (",", "\n", "\r")):
        raise ValueError("unsafe Docker bind path")
    return str(mapped)


def container_command(args, runtime, mounts, *, name):
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "4g",
        "--cpus",
        "2",
        "--pids-limit",
        "256",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=128m",
        "--tmpfs",
        "/work:rw,noexec,nosuid,size=128m,mode=1777",
        "--workdir",
        "/work",
        "--env",
        "HOME=/tmp",
    ]
    for source, target, readonly in [
        (args.author_root, "/original", True),
        (runtime, "/runtime", True),
        *mounts,
    ]:
        mapped = daemon_path(source, args.host_bind_root, args.daemon_bind_root)
        command += [
            "--mount",
            f"type=bind,src={mapped},dst={target}" + (",readonly" if readonly else ""),
        ]
    return command + [args.image, args.python, "-B"]


def tool_once(args, runtime, data, queue, case_output):
    request = queue / "original-python.request.json"
    response = queue / "original-python.response.json"
    claim = queue / "original-python.claim"
    if not request.exists() or response.exists() or claim.exists():
        return
    if request.is_symlink():
        raise ValueError("tool request symlink rejected")
    payload = json.loads(request.read_bytes())
    if not isinstance(payload.get("code"), str) or hashlib.sha256(
        payload["code"].encode()
    ).hexdigest() != payload.get("code_sha256"):
        raise ValueError("original tool request identity differs")
    with claim.open("xb") as stream:
        stream.write(b"claimed-once; do not resubmit")
    name = "discovery-tool-" + uuid.uuid4().hex
    command = container_command(args, runtime, [(data, "/input", True)], name=name)
    command += ["/runtime/discoverybench_isolated_python.py", "--source", "/original"]
    # Only science data, pinned source and public runtime enter this container.
    command.insert(2, "-i")
    try:
        result = subprocess.run(
            command,
            input=json.dumps({"code": payload["code"]}).encode(),
            capture_output=True,
            timeout=args.transport_timeout,
        )
    except subprocess.TimeoutExpired as error:
        subprocess.run(["docker", "kill", name], capture_output=True, check=False)
        (case_output / "tool-stdout-private.log").write_bytes(error.stdout or b"")
        (case_output / "tool-stderr-private.log").write_bytes(error.stderr or b"")
        raise RuntimeError(
            "tool timed out; claim retained, no automatic replay"
        ) from error
    (case_output / "tool-stdout-private.log").write_bytes(result.stdout)
    (case_output / "tool-stderr-private.log").write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError("isolated tool failed; saved stderr and claim retained")
    answer = json.loads(result.stdout)
    if not isinstance(answer.get("tool_output"), str):
        raise ValueError("original tool string output required")
    save_new(
        response,
        json.dumps(
            {
                "code_sha256": payload["code_sha256"],
                "status": "ORIGINAL_TOOL_RETURNED",
                "tool_output": answer["tool_output"],
            }
        ).encode(),
        owner_uid=request.stat().st_uid,
    )


def run_child(args, command, queue, output, *, tool=None):
    with (
        (output / "stdout-private.log").open("xb") as stdout,
        (output / "stderr-private.log").open("xb") as stderr,
    ):
        child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr
        )
        (output / "launch-private.json").write_text(
            json.dumps(
                {
                    "pid": child.pid,
                    "container_name": command[command.index("--name") + 1],
                }
            )
        )
        # On transport errors leave original child/job claims intact for diagnosis.
        # An observation failure never triggers a second model/tool request.
        while child.poll() is None:
            if not args.preflight:
                if not args.external_model_worker:
                    process_one(
                        queue,
                        args.api_base,
                        api_key=os.environ[args.api_key_env]
                        if args.api_key_env
                        else None,
                        disable_thinking=args.disable_thinking,
                        timeout=args.transport_timeout,
                    )
                if tool:
                    tool()
            time.sleep(0.2)
        if child.returncode:
            raise RuntimeError(
                "original child failed; retain logs, do not replace task"
            )


def private_directory(path):
    path.mkdir(mode=0o700)
    os.chown(path, 65534, 65534)


def stage_inputs(prepared, case, output):
    prefix = Path(f"case-{case['original_position']}")
    for item in [case["metadata"], case["reference"], *case["data_files"]]:
        relative = Path(item["path"])
        if not relative.is_relative_to(prefix) or ".." in relative.parts:
            raise ValueError("prepared case file has inconsistent original position")
        within = relative.relative_to(prefix)
        if within.is_absolute():
            raise ValueError("relative original case path required")
        source = checked_file(prepared, item)
        target = output / within
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("staged original file identity differs")
    for path in [output, *output.rglob("*")]:
        path.chmod(0o755 if path.is_dir() else 0o644)


def execute(args):
    if os.name != "posix" or os.geteuid() != 0:
        raise ValueError(
            "owned Linux Docker host with queue ownership privilege required"
        )
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image):
        raise ValueError("immutable local Docker image ID required, not a mutable tag")
    args.host_bind_root = args.host_bind_root.resolve(strict=True)
    args.author_root = verify_author(args.author_root)
    verify_scorer(args.author_root)
    prepared, cases = load_cases(args.cases)
    if args.output.exists() or args.output.resolve().is_relative_to(
        Path(__file__).resolve().parents[1]
    ):
        raise ValueError("fresh output outside the repository required")
    if args.output.resolve().is_relative_to(prepared) or prepared.is_relative_to(
        args.output.resolve()
    ):
        raise ValueError("run output and prepared inputs must be separate")
    if not args.preflight and not args.external_model_worker:
        endpoint = urlsplit(args.api_base)
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint.username
            or endpoint.password
        ):
            raise ValueError("explicit owned loopback model endpoint required")
    daemon = subprocess.check_output(
        ["docker", "info", "--format", "{{.ID}}"], text=True
    ).strip()
    if daemon != args.daemon_id:
        raise ValueError("Docker daemon identity differs")
    args.output.mkdir(parents=True, mode=0o700, exist_ok=False)
    # Parent is traversable only for UID65534 container jobs in this owned run.
    args.output.chmod(0o711)
    runtime = args.output / "runtime"
    runtime.mkdir(mode=0o755)
    runtime.chmod(0o755)
    for name in RUNTIME:
        target = runtime / name
        shutil.copyfile(Path(__file__).parent / name, target)
        target.chmod(0o644)
    completed = []
    for case in cases:
        case_dir = args.output / f"case-{case['original_position']}"
        case_dir.mkdir(mode=0o711)
        for name in (
            "model-queue",
            "tool-queue",
            "actor",
            "judge-model-queue",
            "judge",
        ):
            private_directory(case_dir / name)
        # Copy only verified original inputs, preserving names and reference bytes.
        inputs = case_dir / "inputs"
        stage_inputs(prepared, case, inputs)
        metadata = inputs / "metadata" / Path(case["metadata"]["path"]).name
        data = inputs / "data"
        queue = case_dir / "model-queue"
        tool_queue = case_dir / "tool-queue"
        command = container_command(
            args,
            runtime,
            [
                (metadata.parent, "/metadata", True),
                (data, "/input", True),
                (queue, "/model-queue", False),
                (tool_queue, "/tool-queue", False),
                (case_dir / "actor", "/state", False),
            ],
            name="discovery-actor-" + uuid.uuid4().hex,
        )
        command += [
            "/runtime/run_discoverybench_original_actor.py",
            "--author-root",
            "/original",
            "--metadata",
            "/metadata/" + metadata.name,
            "--metadata-sha256",
            case["metadata"]["sha256"],
            "--query-id",
            str(int(case["query_id"])),
            "--dataset-root",
            "/input",
            "--model",
            args.model,
            "--model-queue",
            "/model-queue",
            "--tool-queue",
            "/tool-queue",
            "--output",
            "/state/run",
            "--transport-timeout",
            str(args.transport_timeout),
            "--preflight" if args.preflight else "--execute",
        ]
        run_child(
            args,
            command,
            queue,
            case_dir / "actor",
            tool=lambda: tool_once(args, runtime, data, tool_queue, case_dir),
        )
        if not args.preflight:
            prediction = case_dir / "actor/run/original-agent-results-private.json"
            scoring = case_dir / "scoring-input"
            scoring.mkdir(mode=0o755)
            for source, name in [
                (metadata, metadata.name),
                (inputs / "reference.json", "reference.json"),
                (prediction, "prediction.json"),
            ]:
                shutil.copyfile(source, scoring / name)
                (scoring / name).chmod(0o644)
            judge_queue = case_dir / "judge-model-queue"
            command = container_command(
                args,
                runtime,
                [
                    (scoring, "/input", True),
                    (judge_queue, "/model-queue", False),
                    (case_dir / "judge", "/state", False),
                ],
                name="discovery-judge-" + uuid.uuid4().hex,
            )
            command += [
                "/runtime/run_discoverybench_original_judge.py",
                "--author-root",
                "/original",
                "--model",
                args.judge_model,
                "--model-queue",
                "/model-queue",
                "--output",
                "/state/run",
                "--transport-timeout",
                str(args.transport_timeout),
                "--execute",
            ]
            for name in ("metadata", "reference", "prediction"):
                filename = metadata.name if name == "metadata" else name + ".json"
                command += [
                    "--" + name,
                    f"/input/{filename}",
                    "--" + name + "-sha256",
                    hashlib.sha256((scoring / filename).read_bytes()).hexdigest(),
                ]
            run_child(args, command, judge_queue, case_dir / "judge")
        completed.append(case["original_position"])
        (args.output / "progress-public.json").write_text(
            json.dumps(
                {
                    "completed_positions": completed,
                    "preflight_only": args.preflight,
                    "source_checkbox_eligible": False,
                }
            )
        )
    print(
        json.dumps(
            {
                "status": "OFFLINE_ACTOR_PREFLIGHT_COMPLETED"
                if args.preflight
                else "ORIGINAL_ACTOR_TOOL_SCORER_RETURNED_AUDIT_PENDING",
                "completed_positions": completed,
                "source_checkbox_eligible": False,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cases",
        "author-root",
        "host-bind-root",
        "daemon-bind-root",
        "output",
    ):
        parser.add_argument("--" + name, type=Path)
    for name in ("image", "daemon-id", "api-base", "api-key-env"):
        parser.add_argument("--" + name)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--judge-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--python", default="/opt/discovery-client/bin/python")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument(
        "--external-model-worker",
        action="store_true",
        help="operator serves dedicated model queues externally; no local HTTP forwarding",
    )
    parser.add_argument("--transport-timeout", type=float, default=3600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.preflight and not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "model_calls": 0}))
        return
    required = (
        "cases",
        "author_root",
        "host_bind_root",
        "daemon_bind_root",
        "output",
        "image",
        "daemon_id",
    )
    if (
        any(getattr(args, key) is None for key in required)
        or args.transport_timeout <= 0
        or (args.execute and bool(args.api_base) == args.external_model_worker)
    ):
        parser.error(
            "explicit prepared cases, owned immutable Docker environment, fresh output and model endpoint required"
        )
    os.umask(0o077)
    execute(args)


if __name__ == "__main__":
    main()
