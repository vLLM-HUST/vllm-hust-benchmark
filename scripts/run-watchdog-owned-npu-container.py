#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence

from vllm_hust_benchmark.runner_ownership import (
    build_docker_create_command,
    resolve_runner_device,
    validate_container_inspect,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one watchdog-owned NPU container from a poy-180 runner."
    )
    parser.add_argument("--runner-name", default=os.environ.get("RUNNER_NAME", ""))
    parser.add_argument("--name", required=True, help="Unique Docker container name")
    parser.add_argument("--image", required=True)
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--keep-container", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        assignment = resolve_runner_device(args.runner_name)
        create_command = build_docker_create_command(
            assignment=assignment,
            container_name=args.name,
            image=args.image,
            command=command,
            volumes=args.volume,
            extra_env=args.env,
        )
    except ValueError as exc:
        print(f"preflight failed: {exc}", file=sys.stderr)
        return 2

    print(
        "runner ownership: "
        f"runner={assignment.runner_name} "
        f"physical={assignment.physical_device} "
        f"container-logical={assignment.logical_device}"
    )
    container_id = ""
    try:
        created = subprocess.run(
            create_command,
            check=True,
            capture_output=True,
            text=True,
        )
        container_id = created.stdout.strip()
        if not container_id:
            raise RuntimeError("docker create returned an empty container ID")

        inspected = subprocess.run(
            ["docker", "inspect", container_id],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(inspected.stdout)
        if len(payload) != 1:
            raise RuntimeError("docker inspect did not return exactly one container")
        validate_container_inspect(payload[0], assignment)
        print("watchdog ownership preflight: ok")

        subprocess.run(["docker", "start", container_id], check=True)
        wait_result = subprocess.run(
            ["docker", "wait", container_id],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(["docker", "logs", container_id], check=False)
        return int(wait_result.stdout.strip())
    except (
        json.JSONDecodeError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"container execution failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if container_id and not args.keep_container:
            subprocess.run(
                ["docker", "rm", "--force", container_id],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    raise SystemExit(run())
