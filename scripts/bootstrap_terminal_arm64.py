#!/usr/bin/env python3
"""Prepare and run one native ARM64 Terminal-Bench task with upstream Harbor.

Requires a user-owned Docker workspace, Python >=3.12 and an existing model API.
No host/cluster administration, model deployment, benchmark answer or custom scorer.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

TASK = "openssl-selfsigned-cert"
REV = "7131e4375048a0e408a8fb404b5f499d726b695b"  # pragma: allowlist secret — public source identity
HARBOR = "0.24.0"
BASE_MANIFEST = "608b27b2db85abf38d81685085cd1166a5ed7ada66549f0dfe53a75bb830cd5d"  # pragma: allowlist secret — public source identity
BASE_CONFIG = "18d58f4e045f133283c50016a09285439cbe765877e1a93760888b85fd2ce905"  # pragma: allowlist secret — public source identity
FILES = {
    "instruction.md": "38054a3371f7c2c853478a2f9af0bff858437d5b28a9a07483238f06e566921f",  # pragma: allowlist secret — public source identity
    "task.toml": "75c8913f0fa15890786c11999cde920b8bfc10d4ce8a14edf1e76985a2ac5800",  # pragma: allowlist secret — public source identity
    "environment/Dockerfile": "f590b523038be43afcaf5291d7b9e8ed1333c48d9994e77b8dee2f2c352f0f40",  # pragma: allowlist secret — public source identity
    "tests/test.sh": "4770437ea96c3cc84684b4f99d55fb148fcac09f9ea1e8ef49de487716e6c334",  # pragma: allowlist secret — public source identity
    "tests/test_outputs.py": "90970bb3c372e8e7fb6bb9ac5f649ba73eb38e1a5ea2e922a9057a72b6b4192d",  # pragma: allowlist secret — public source identity
    "solution/solve.sh": "58cf3c844e907745cd81fb4bd75efb3d080e2f0e82b589c56c954ae6e1756510",  # pragma: allowlist secret — public source identity
}
UV = {
    "uv-installer.sh": "8402ab80d2ef54d7044a71ea4e4e1e8db3b20c87c7bffbc30bff59f1e80ebbd5",  # pragma: allowlist secret — public source identity
    "uv-aarch64-unknown-linux-gnu.tar.gz": "9db0c2f6683099f86bfeea47f4134e915f382512278de95b2a0e625957594ff3",  # pragma: allowlist secret — public source identity
}
COMPOSE_SHA = "49082844b87f03cdcd5f5bbef1ba8c9c897b7a2dfb80cea18d61ec8ca6117e0c"  # pragma: allowlist secret — public source identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def fetch(url, binary=False):
    headers = {"User-Agent": "vllm-hust-terminal-arm64"}
    if binary:
        headers["Accept"] = "application/octet-stream"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=120
    ) as r:
        return r.read()


def checked_download(url, target, digest, binary=False):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        data = fetch(url, binary)
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Download hash mismatch: {target.name}")
        with target.open("xb") as f:
            f.write(data)
    if sha(target) != digest:
        raise RuntimeError(f"Existing artifact hash mismatch: {target.name}")


def command(argv, env, logfile=None, timeout=900):
    if logfile:
        with Path(logfile).open("ab") as stream:
            result = subprocess.run(
                argv,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(
                f"Command failed ({result.returncode}); inspect {logfile}"
            )
        return ""
    return subprocess.check_output(argv, env=env, text=True, timeout=timeout)


def environment(root):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(
        HOME=str(root / "home"),
        XDG_CACHE_HOME=str(root / "cache"),
        TMPDIR=str(root / "tmp"),
        DOCKER_CONFIG=str(root / "docker-config"),
        UV_CACHE_DIR=str(root / "uv-cache"),
        NO_COLOR="1",
        PYTHONDONTWRITEBYTECODE="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
    )
    if not env.get("DOCKER_HOST"):
        for path in ("/run/swe-pro-docker/docker.sock", "/var/run/docker.sock"):
            if Path(path).is_socket():
                env["DOCKER_HOST"] = "unix://" + path
                break
    return env


def docker(env, *args):
    return command(["docker", *args], env, timeout=120)


def verify_preparation(root):
    state = json.loads((root / "prepared.json").read_text())
    for relative, digest in state["bindings"].items():
        if sha(root / relative) != digest:
            raise RuntimeError(f"Prepared source changed: {relative}")
    image = json.loads(
        docker(environment(root), "image", "inspect", state["image_id"])
    )[0]
    if image["Id"] != state["image_id"] or image["Architecture"] != "arm64":
        raise RuntimeError("Image identity or native ARM64 architecture changed")
    return state


def prepare(root, cpuset, resume_prepare=False, base_image_ref=None):
    base_image_ref = base_image_ref or "python@sha256:" + BASE_MANIFEST
    if not base_image_ref.endswith("@sha256:" + BASE_MANIFEST):
        raise ValueError("Base image reference must use the pinned ARM64 manifest")
    if platform.machine() not in ("aarch64", "arm64") or sys.version_info < (3, 12):
        raise RuntimeError("Use a native ARM64 CPU workspace with Python >=3.12")
    if (root / "prepared.json").exists():
        verify_preparation(root)
        print("READY: existing preparation verified; no task rerun")
        return
    if not root.exists():
        root.mkdir(parents=True, mode=0o700)
    for name in (
        "home",
        "cache",
        "tmp",
        "uv-cache",
        "docker-config",
        "downloads",
        "jobs",
    ):
        (root / name).mkdir(exist_ok=True, mode=0o700)
    env = environment(root)
    if shutil.disk_usage(root).free < 5 * 2**30:
        raise RuntimeError(
            "Preparation needs at least 5 GiB free in the chosen persistent directory"
        )
    info = json.loads(docker(env, "info", "--format", "{{json .}}"))
    if info["Architecture"] not in ("aarch64", "arm64"):
        raise RuntimeError("Docker server is not ARM64")
    plugin = root / "docker-config/cli-plugins/docker-compose"
    checked_download(
        "https://api.github.com/repos/docker/compose/releases/assets/294974310",
        plugin,
        COMPOSE_SHA,
        True,
    )
    plugin.chmod(0o755)
    compose_version = docker(env, "compose", "version").strip()
    for name, digest in FILES.items():
        p = root / "original" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            response = json.loads(
                fetch(
                    f"https://api.github.com/repos/harbor-framework/terminal-bench-2-1/contents/tasks/{TASK}/{name}?ref={REV}"
                )
            )
            data = base64.b64decode(response["content"])
            if hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError(f"Upstream task identity mismatch: {name}")
            p.write_bytes(data)
        if sha(p) != digest:
            raise RuntimeError(f"Task hash mismatch: {name}")
        p.chmod(0o755 if name.endswith(".sh") else 0o644)
    release = json.loads(
        fetch("https://api.github.com/repos/astral-sh/uv/releases/tags/0.9.5")
    )
    for name, digest in UV.items():
        asset = next(a for a in release["assets"] if a["name"] == name)
        if asset.get("digest") != "sha256:" + digest:
            raise RuntimeError("Pinned uv release digest mismatch")
        checked_download(asset["url"], root / "downloads" / name, digest, True)
    client = root / "client"
    if not client.exists():
        command([sys.executable, "-m", "venv", str(client)], env, root / "install.log")
    command(
        [str(client / "bin/python"), "-m", "pip", "install", "harbor==" + HARBOR],
        env,
        root / "install.log",
        900,
    )
    (root / "requirements-installed.txt").write_text(
        command([str(client / "bin/python"), "-m", "pip", "freeze"], env)
    )
    # A normal immutable base cache is allowed; never reuse an answer-bearing image.
    base_id = "sha256:" + BASE_CONFIG
    try:
        base = json.loads(docker(env, "image", "inspect", base_id))[0]
    except subprocess.CalledProcessError:
        command(
            [
                "docker",
                "pull",
                "--platform",
                "linux/arm64",
                base_image_ref,
            ],
            env,
            root / "base-pull.log",
            900,
        )
        base = json.loads(docker(env, "image", "inspect", base_id))[0]
    if base["Id"] != base_id or base["Architecture"] != "arm64":
        raise RuntimeError("Base image mismatch")
    base_tag = "terminal-user-base:" + BASE_MANIFEST[:12]
    docker(env, "tag", base_id, base_tag)
    task = root / "task"
    if task.exists():
        if not resume_prepare or list(root.glob("*-started.json")):
            raise RuntimeError(
                "Partial preparation retained; --resume-prepare is only allowed before any trial"
            )
        for name, digest in FILES.items():
            if (
                name not in ("task.toml", "environment/Dockerfile")
                and sha(task / name) != digest
            ):
                raise RuntimeError("Partial task input changed")
    else:
        shutil.copytree(root / "original", task)
    build = task / "environment"
    for name in UV:
        shutil.copyfile(root / "downloads" / name, build / name)
    (build / "curl").write_text("""#!/bin/sh
if [ "$#" = 2 ] && [ "$1" = '-LsSf' ] && [ "$2" = 'https://astral.sh/uv/0.9.5/install.sh' ]; then
    exec /bin/cat /opt/verified-uv/uv-installer.sh
fi
exec /usr/bin/curl "$@"
""")
    dockerfile = (root / "original/environment/Dockerfile").read_text()
    dockerfile = dockerfile.replace(
        "FROM python:3.13-slim-bookworm", "FROM " + base_tag
    )
    dockerfile += """
# Dependency transport only; upstream instructions, solution and verifier stay unchanged.
RUN apt-get update && apt-get install -y curl tmux asciinema && rm -rf /var/lib/apt/lists/*
COPY uv-installer.sh uv-aarch64-unknown-linux-gnu.tar.gz /opt/verified-uv/
COPY curl /opt/verified-uv/bin/curl
RUN chmod 755 /opt/verified-uv/bin/curl && UV_DOWNLOAD_URL=file:///opt/verified-uv sh /opt/verified-uv/uv-installer.sh
ENV PATH="/opt/verified-uv/bin:/root/.local/bin:${PATH}"
ENV UV_DOWNLOAD_URL="file:///opt/verified-uv"
ENV UV_CACHE_DIR="/opt/verified-uv/cache"
RUN uvx -p 3.13 -w pytest==8.4.1 -w pytest-json-ctrf==0.3.5 pytest --version
"""
    (build / "Dockerfile").write_text(dockerfile)
    tag = (
        "terminal-user-openssl:" + hashlib.sha256(dockerfile.encode()).hexdigest()[:16]
    )
    network = "terminal-user-" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
    existing_networks = docker(
        env, "network", "ls", "--format", "{{.Name}}"
    ).splitlines()
    if network not in existing_networks:
        docker(
            env,
            "network",
            "create",
            "--label",
            "vllm-hust.terminal-user=true",
            "--opt",
            "com.docker.network.driver.mtu=1400",
            network,
        )
    network_info = json.loads(docker(env, "network", "inspect", network))[0]
    if (
        network_info["Labels"].get("vllm-hust.terminal-user") != "true"
        or network_info["Options"].get("com.docker.network.driver.mtu") != "1400"
    ):
        raise RuntimeError("Owned build/task network identity mismatch")
    build_argv = [
        "docker",
        "build",
        "--network",
        network,
        "--memory",
        "2g",
        "--cpu-period",
        "100000",
        "--cpu-quota",
        "100000",
        "--platform",
        "linux/arm64",
        "-t",
        tag,
    ]
    if cpuset:
        build_argv += ["--cpuset-cpus", cpuset]
    command(build_argv + [str(build)], env, root / "build.log", 900)
    image = json.loads(docker(env, "image", "inspect", tag))[0]
    toml = (task / "task.toml").read_text()
    toml, count = re.subn(
        r"^docker_image\s*=.*$",
        'docker_image = "' + tag + '"',
        toml,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Unexpected upstream task image declaration")
    (task / "task.toml").write_text(toml)
    main = {
        "platform": "linux/arm64",
        "pull_policy": "never",
        "cpus": 1.0,
        "mem_limit": "2048m",
        "memswap_limit": "2048m",
        "networks": ["default"],
    }
    if cpuset:
        main["cpuset"] = cpuset
    save(
        root / "compose.json",
        {
            "services": {"main": main},
            "networks": {"default": {"external": True, "name": network}},
        },
    )
    # Verify that the daemon and client see the same chosen persistent directory.
    probe = hashlib.sha256(os.urandom(32)).hexdigest()
    (root / "mount-probe.txt").write_text(probe)
    actual = docker(
        env,
        "run",
        "--rm",
        "--network",
        network,
        "--cpus",
        "1",
        "--memory",
        "2g",
        "-v",
        str(root) + ":/probe",
        tag,
        "sh",
        "-c",
        "cat /probe/mount-probe.txt; printf verified > /probe/mount-reply.txt",
    )
    if actual != probe or (root / "mount-reply.txt").read_text() != "verified":
        raise RuntimeError("Docker daemon/client persistent path visibility mismatch")
    bindings = {
        str(p.relative_to(root)): sha(p) for p in sorted(task.rglob("*")) if p.is_file()
    }
    bindings["compose.json"] = sha(root / "compose.json")
    bindings["requirements-installed.txt"] = sha(root / "requirements-installed.txt")
    save(
        root / "prepared.json",
        {
            "schema": "terminal-arm64-user-preparation/v1",
            "task": TASK,
            "task_revision": REV,
            "harbor_version": HARBOR,
            "architecture": image["Architecture"],
            "base_manifest": "sha256:" + BASE_MANIFEST,
            "base_image_ref": base_image_ref,
            "base_image_id": base_id,
            "image_id": image["Id"],
            "image_tag": tag,
            "network": network,
            "compose_version": compose_version,
            "bindings": bindings,
            "task_cpu_limit": 1,
            "task_memory_bytes": 2147483648,
            "disk_limit_enforced": False,
            "fresh_client": True,
            "daemon_client_mount_probe": "passed",
            "prepared_epoch": time.time(),
        },
    )
    print("READY: one ARM64 task prepared; model execution and scoring not yet run")


def config(root, role, profile):
    agent = {"name": role}
    if role == "model":
        agent = {
            "name": "terminus-2",
            "model_name": "openai/" + profile["model"],
            "kwargs": {
                "api_base": profile["api_base"],
                "max_turns": 100,
                "temperature": 0,
                "enable_summarize": False,
                "interleaved_thinking": False,
                "proactive_summarization_threshold": 0,
                "parser_name": "json",
                "record_terminal_session": True,
                "store_all_messages": True,
                "use_responses_api": False,
                "llm_kwargs": {"num_retries": 0},
                "llm_call_kwargs": {
                    "max_tokens": 4096,
                    "seed": 17,
                    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                },
                "model_info": {
                    "max_input_tokens": 32768,
                    "max_output_tokens": 4096,
                    "max_tokens": 32768,
                    "input_cost_per_token": 0,
                    "output_cost_per_token": 0,
                },
            },
        }
    return {
        "job_name": "terminal-user-" + role,
        "jobs_dir": str(root / "jobs"),
        "n_attempts": 1,
        "n_concurrent_trials": 1,
        "retry": {"max_retries": 0},
        "environment": {
            "type": "docker",
            "delete": True,
            "force_build": False,
            "cpu_enforcement_policy": "limit",
            "memory_enforcement_policy": "limit",
            "extra_docker_compose": [str(root / "compose.json")],
        },
        "agents": [agent],
        "tasks": [{"path": str(root / "task")}],
    }


def run_once(root, role, profile=None):
    if (root / (role + "-started.json")).exists():
        raise RuntimeError(
            "This attempt already started; inspect its receipt instead of rerunning"
        )
    state = verify_preparation(root)
    env = environment(root)
    if role == "model":
        for control, reward in (("nop", 0), ("oracle", 1)):
            previous = json.loads((root / (control + "-receipt.json")).read_text())
            if previous["reward"] != reward or previous["tests"] != 6:
                raise RuntimeError("Verifier controls have not passed")
        if not profile:
            raise RuntimeError("Provide --model-api and --model-name, or --profile")
        if profile.get("api_key_file"):
            key = Path(profile["api_key_file"])
            if key.stat().st_mode & 0o077:
                raise RuntimeError("Model credential file must be private")
            env["OPENAI_API_KEY"] = key.read_text().strip()
        elif profile.get("no_api_key"):
            env["OPENAI_API_KEY"] = (
                "not-required"  # pragma: allowlist secret - non-secret test or unauthenticated-service placeholder
            )
        elif not env.get(profile.get("api_key_env", "OPENAI_API_KEY")):
            raise RuntimeError("Model API credential environment variable is missing")
        else:
            env["OPENAI_API_KEY"] = env[profile.get("api_key_env", "OPENAI_API_KEY")]
    cfg = config(root, role, profile)
    job_path = root / (role + "-job.json")
    if job_path.exists():
        if json.loads(job_path.read_text()) != cfg:
            raise RuntimeError("Frozen job differs from requested configuration")
    else:
        save(job_path, cfg)
    save(
        root / (role + "-started.json"),
        {
            "epoch": time.time(),
            "job_sha256": sha(root / (role + "-job.json")),
            "image_id": state["image_id"],
            "attempt_limit": 1,
            "automatic_whole_trial_retries": 0,
            "harbor_internal_call_retries": "upstream defaults; SDK num_retries=0",
            "role": role,
        },
    )
    logfile = root / (role + ".log")
    code = None
    try:
        with logfile.open("xb") as log:
            p = subprocess.Popen(
                [
                    str(root / "client/bin/harbor"),
                    "run",
                    "-c",
                    str(root / (role + "-job.json")),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = p.wait(timeout=2400)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
                raise
    finally:
        results = list((root / "jobs" / cfg["job_name"]).glob("*/result.json"))
        reward = None
        exception = None
        tests = passed = None
        if len(results) == 1:
            result = json.loads(results[0].read_text())
            reward = (
                (result.get("verifier_result") or {}).get("rewards", {}).get("reward")
            )
            exception = (result.get("exception_info") or {}).get("exception_type")
            ctrf = results[0].parent / "verifier/ctrf.json"
            if ctrf.exists():
                summary = json.loads(ctrf.read_text())["results"]["summary"]
                tests, passed = summary["tests"], summary["passed"]
        save(
            root / (role + "-receipt.json"),
            {
                "role": role,
                "process_exit_code": code,
                "reward": reward,
                "tests": tests,
                "passed": passed,
                "exception_type": exception,
                "result_paths": [str(p) for p in results],
                "ended_epoch": time.time(),
                "status": "scored" if reward is not None else "not_scored",
            },
        )
    print(json.dumps(json.loads((root / (role + "-receipt.json")).read_text())))
    if reward is None or tests != 6 or code != 0:
        raise RuntimeError(
            f"{role}: incomplete execution; original logs retained; no automatic rerun"
        )


def model_profile(args):
    if args.profile:
        if args.model_api or args.model_name or args.api_key_env or args.no_api_key:
            raise ValueError("Use either --profile or the model connection options")
        profile = json.loads(args.profile.expanduser().read_text())
    else:
        if not args.model_api or not args.model_name:
            raise ValueError("Provide both --model-api and --model-name")
        profile = {
            "api_base": args.model_api.rstrip("/"),
            "model": args.model_name.removeprefix("openai/"),
        }
        if args.no_api_key:
            if args.api_key_env:
                raise ValueError("Use either --api-key-env or --no-api-key")
            profile["no_api_key"] = True
        else:
            profile["api_key_env"] = args.api_key_env or "OPENAI_API_KEY"
    url = urlsplit(profile["api_base"])
    if url.scheme not in ("http", "https") or not url.hostname:
        raise ValueError("Model API must be an HTTP(S) base URL, including /v1")
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("Keep API credentials in an environment variable or key file")
    if not isinstance(profile.get("model"), str) or not profile["model"].strip():
        raise ValueError("Model name is required")
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "controls", "run", "status"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--profile", type=Path, help="Optional private model connection JSON"
    )
    parser.add_argument(
        "--model-api", help="OpenAI-compatible API base URL including /v1"
    )
    parser.add_argument("--model-name", help="Served model name")
    parser.add_argument(
        "--base-image-ref",
        help="Registry mirror reference with the pinned base manifest",
    )
    parser.add_argument(
        "--api-key-env", help="API key environment variable (default OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--no-api-key",
        action="store_true",
        help="Use a service without API authentication",
    )
    parser.add_argument(
        "--resume-prepare",
        action="store_true",
        help="Resume diagnosed preparation failure before any task starts",
    )
    parser.add_argument(
        "--cpuset", help="Optional workspace CPU affinity, provided by ops"
    )
    args = parser.parse_args()
    os.umask(0o077)
    root = args.directory.expanduser().resolve()
    if args.action == "prepare":
        prepare(root, args.cpuset, args.resume_prepare, args.base_image_ref)
    elif args.action == "controls":
        for role in ("nop", "oracle"):
            run_once(root, role)
    elif args.action == "run":
        run_once(
            root,
            "model",
            model_profile(args),
        )
    else:
        print(
            json.dumps(
                {
                    p.name: json.loads(p.read_text())
                    for p in root.glob("*-receipt.json")
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
