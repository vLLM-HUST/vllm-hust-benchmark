"""Run original BigCodeBench scoring in an existing immutable local container image.

No image pull/build, inference or host-execution fallback. Linux root is required
to stage private inputs for a nonzero container UID. Retain container and logs on
failure for inspection; never restart or remove someone else's container.
"""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from prepare_bigcodebench_scoring_plan import build_plan


def container_command(engine, image, name, work, uid, environment):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("immutable locally installed image ID required")
    if not 1 <= uid <= 2**31 - 2:
        raise ValueError("nonzero sandbox UID required")
    work = Path(work)
    if "," in str(work):
        raise ValueError("mount paths containing commas unsupported")
    command = [
        engine,
        "run",
        "--name",
        name,
        "--pull=never",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--user",
        f"{uid}:{uid}",
        "--pids-limit",
        "1024",
        "--memory",
        "32g",
        "--memory-swap",
        "32g",
        "--ulimit",
        "as=32212254720:32212254720",
        "--ulimit",
        "data=32212254720:32212254720",
        "--ulimit",
        "stack=10485760:10485760",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m,mode=1777",
        "--shm-size",
        "64m",
        "--workdir",
        "/output",
        "--entrypoint",
        "/usr/local/bin/python3",
    ]
    for source, target, readonly in (
        ("inputs", "/inputs", True),
        ("author", "/author", True),
        ("output", "/output", False),
    ):
        command += [
            "--mount",
            f"type=bind,source={work / source},target={target}"
            + (",readonly" if readonly else ""),
        ]
    for key, value in environment.items():
        command += ["--env", f"{key}={value}"]
    return [*command, image, "/inputs/worker.py", "--evaluate"]


def launch(args):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("authorized Linux root staging required; no host fallback")
    engine = shutil.which(args.engine)
    if not engine:
        raise RuntimeError("container engine unavailable; no host fallback")
    plan = build_plan(args.prepared_dir, args.official_code)
    work = args.work_root.resolve()
    for source in (args.prepared_dir, args.official_code):
        source = source.resolve()
        if work.is_relative_to(source) or source.is_relative_to(work):
            raise ValueError("fresh work-root must not overlap input/source")
    name = "hust-bcb-" + uuid.uuid4().hex
    command = container_command(
        engine, args.image, name, work, args.sandbox_uid, plan["required_environment"]
    )
    inspected = subprocess.run(
        [engine, "image", "inspect", args.image],
        check=True,
        capture_output=True,
        text=True,
    )
    images = json.loads(inspected.stdout)
    if len(images) != 1 or images[0].get("Id") != args.image:
        raise RuntimeError("local immutable image identity mismatch")
    work.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name_dir in ("inputs", "author", "output"):
        (work / name_dir).mkdir(mode=0o755)
    os.chown(work / "output", args.sandbox_uid, args.sandbox_uid)
    (work / "output").chmod(0o700)
    # Only verified publisher sources and requirements, never the whole host tree.
    for relative in plan["publisher_source_checksums"]:
        target = work / "author" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.official_code / relative, target)
        target.chmod(0o644)
    for directory in (work / "author").rglob("*"):
        if directory.is_dir():
            directory.chmod(0o755)
    shutil.copyfile(
        args.prepared_dir / "original-dataset-private.jsonl",
        work / "inputs/original-dataset-private.jsonl",
    )
    (work / "inputs/original-dataset-private.jsonl").chmod(0o444)
    shutil.copyfile(
        args.prepared_dir / "raw-solutions-private.jsonl",
        work / "output/raw-solutions-private.jsonl",
    )
    os.chown(
        work / "output/raw-solutions-private.jsonl", args.sandbox_uid, args.sandbox_uid
    )
    (work / "output/raw-solutions-private.jsonl").chmod(0o600)
    shutil.copyfile(
        Path(__file__).with_name("bigcodebench_container_worker.py"),
        work / "inputs/worker.py",
    )
    host_namespaces = {
        n: os.readlink(f"/proc/self/ns/{n}") for n in ("user", "mnt", "net", "pid")
    }
    (work / "inputs/control-private.json").write_text(
        json.dumps({"plan": plan, "host_namespaces": host_namespaces})
    )
    for path in (work / "inputs").iterdir():
        path.chmod(0o444)
    control = {
        "source_id": "BIGCODEBENCH",
        "container_name": name,
        "image_id": args.image,
        "status": "STARTING_NOT_SCORED",
        "source_checkbox_eligible": False,
    }
    control_path = work / "launch-private.json"
    control_path.write_text(json.dumps(control, indent=2))
    with (work / "container-private.log").open("xb") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    control["container_cli_exit_code"] = completed.returncode
    receipt = work / "output/receipt-public.json"
    if completed.returncode == 0 and receipt.is_file():
        control["status"] = "CONTAINER_EXITED_REVIEW_NATIVE_RECEIPT"
    else:
        control["status"] = "CONTAINER_FAILED_NOT_A_MODEL_QUALITY_RESULT"
    control_path.write_text(json.dumps(control, indent=2))
    print(json.dumps(control))
    if completed.returncode or not receipt.is_file():
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["docker", "podman"], default="docker")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prepared-dir", required=True, type=Path)
    parser.add_argument("--official-code", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--sandbox-uid", required=True, type=int)
    launch(parser.parse_args())


if __name__ == "__main__":
    main()
