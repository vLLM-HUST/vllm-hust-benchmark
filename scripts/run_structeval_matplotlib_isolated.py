"""Bound one original renderer component on an explicitly owned Docker service.

No model calls, source rewriting, native generation, parser substitution or VQA.
Retains containers/volumes and private artifacts; never deletes or prunes them.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from run_structeval_matplotlib_worker import RENDERERS, verify_inputs

OWNER_LABEL = "vllm-hust.structeval-operation"


def docker(*args):
    return subprocess.check_output(
        ["docker", *map(str, args)], text=True, timeout=60
    ).strip()


def worker_command(
    image,
    operation,
    software,
    inputs,
    task_id,
    code_sha,
    outputs,
    renderer_kind="matplotlib",
):
    return [
        "run",
        "-d",
        "--pull=never",
        "--name",
        operation + "-worker",
        "--label",
        OWNER_LABEL + "=" + operation,
        "--network",
        "none",
        "--user",
        "65534:65534",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "2g",
        "--cpus",
        "1",
        "--pids-limit",
        "256",
        "--tmpfs",
        "/tmp:size=256m",
        "--log-driver",
        "json-file",
        "--log-opt",
        "max-size=8m",
        "--log-opt",
        "max-file=2",
        "-v",
        software + ":/software:ro",
        "-v",
        inputs + ":/input:ro",
        "-v",
        outputs + ":/output",
        "--entrypoint",
        "/opt/venv/bin/python",
        image,
        "-B",
        "/software/worker.py",
        "--task-id",
        task_id,
        "--renderer",
        "/software/render_matplotlib.py",
        "--code-file",
        "/input/parsed-code.txt",
        "--code-sha256",
        code_sha,
        "--output",
        "/output/job",
        "--execute",
        "--renderer-kind",
        renderer_kind,
    ]


def stop_owned(identity, operation, role="worker"):
    actual = json.loads(docker("inspect", identity))[0]
    if (
        actual["Id"] != identity
        or actual["Config"].get("Labels", {}).get(OWNER_LABEL) != operation
        or role not in {"worker", "export"}
        or actual["Name"] != "/" + operation + "-" + role
    ):
        raise ValueError("refuse to stop a container without exact owned identity")
    docker("stop", "--time", "5", identity)


def export_file(identity, source, destination, maximum):
    docker("cp", identity + ":" + source, destination)
    if (
        destination.is_symlink()
        or not destination.is_file()
        or destination.stat().st_size > maximum
    ):
        raise ValueError("exported component artifact is not a bounded regular file")


def run(args):
    kind = getattr(args, "renderer_kind", "matplotlib")
    if (
        kind not in RENDERERS
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.task_id)
        or not 1 <= args.timeout <= 600
    ):
        raise ValueError(
            "explicit content-addressed image, safe task ID and bounded timeout required"
        )
    result = {
        "source_id": "STRUCTEVAL",
        "status": "PLAN_ONLY",
        "task_id": args.task_id,
        "model_calls": 0,
        "native_pipeline_complete": False,
        "vqa_executed": False,
        "source_checkbox_eligible": False,
    }
    if not args.execute:
        return result
    if sys.platform != "linux" or not shutil.which("docker"):
        raise ValueError("explicitly owned Linux Docker service required")
    os.umask(0o077)
    renderer, code_file, _ = verify_inputs(
        args.renderer, args.code_file, args.code_sha256, kind
    )
    output = args.output.resolve()
    repository = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or not output.parent.is_dir()
        or output.is_relative_to(repository)
    ):
        raise ValueError("fresh external private output with existing parent required")
    if shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise ValueError("less than2GiB free output space; preserve existing files")
    image = json.loads(docker("image", "inspect", args.image))[0]
    if (
        image["Id"] != args.image
        or image["Config"].get("Volumes")
        or image["Config"].get("Healthcheck")
    ):
        raise ValueError(
            "qualified local image without implicit volumes/healthcheck required"
        )
    output.mkdir(mode=0o700)
    operation = "structeval-mpl-" + uuid.uuid4().hex
    software, inputs, outputs = (
        operation + "-software",
        operation + "-input",
        operation + "-output",
    )
    result.update(
        status="PREPARING_PRIVATE_COMPONENT",
        operation=operation,
        image=args.image,
        timeout_seconds=args.timeout,
        parsed_code_sha256=args.code_sha256,
        renderer_kind=kind,
    )

    def save():
        temporary = output / "controller-receipt.partial"
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output / "controller-receipt.json")

    save()
    try:
        for name in (software, inputs):
            docker("volume", "create", "--label", OWNER_LABEL + "=" + operation, name)
        docker(
            "volume",
            "create",
            "--label",
            OWNER_LABEL + "=" + operation,
            "--driver",
            "local",
            "--opt",
            "type=tmpfs",
            "--opt",
            "device=tmpfs",
            "--opt",
            "o=size=256m,uid=65534,gid=65534,mode=0700",
            outputs,
        )
        # Trusted ownership-only staging. Never imports or evaluates input code;
        # changes permissions only inside these new dedicated named volumes.
        ownership = "import os; roots=['/software','/input'];\nfor root in roots:\n for item in os.scandir(root):\n  assert item.is_file(follow_symlinks=False); os.chown(item.path,65534,65534); os.chmod(item.path,0o400)\n os.chown(root,65534,65534); os.chmod(root,0o500)"
        stage = docker(
            "create",
            "--pull=never",
            "--name",
            operation + "-stage",
            "--label",
            OWNER_LABEL + "=" + operation,
            "--network",
            "none",
            "--user",
            "0:0",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "CHOWN",
            "--cap-add",
            "FOWNER",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--pids-limit",
            "32",
            "-v",
            software + ":/software",
            "-v",
            inputs + ":/input",
            "--entrypoint",
            "/opt/venv/bin/python",
            args.image,
            "-B",
            "-c",
            ownership,
        )
        docker(
            "cp",
            Path(__file__).with_name("run_structeval_matplotlib_worker.py"),
            stage + ":/software/worker.py",
        )
        docker("cp", renderer, stage + ":/software/render_matplotlib.py")
        docker("cp", code_file, stage + ":/input/parsed-code.txt")
        docker("start", "-a", stage)
        if json.loads(docker("inspect", stage))[0]["State"]["ExitCode"] != 0:
            raise ValueError("owned metadata-only staging failed")
        holder_code = "import signal,sys,time; signal.signal(signal.SIGTERM,lambda *_:sys.exit(0)); time.sleep(3600)"
        holder = docker(
            "run",
            "-d",
            "--pull=never",
            "--name",
            operation + "-export",
            "--label",
            OWNER_LABEL + "=" + operation,
            "--network",
            "none",
            "--user",
            "65534:65534",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "64m",
            "--cpus",
            "0.1",
            "--pids-limit",
            "32",
            "-v",
            outputs + ":/output:ro",
            "--entrypoint",
            "/opt/venv/bin/python",
            args.image,
            "-B",
            "-c",
            holder_code,
        )
        result.update(
            exporter_container_id=holder,
            output_volume=outputs,
            output_tmpfs_capacity_bytes=256 * 1024**2,
            exporter_lease_seconds=3600,
        )
        save()
        identity = docker(
            *worker_command(
                args.image,
                operation,
                software,
                inputs,
                args.task_id,
                args.code_sha256,
                outputs,
                kind,
            )
        )
        result.update(
            status="ORIGINAL_RENDERER_COMPONENT_RUNNING", container_id=identity
        )
        save()
        try:
            waited = subprocess.run(
                ["docker", "wait", identity],
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=True,
            )
        except subprocess.TimeoutExpired:
            stop_owned(identity, operation)
            result.update(
                status="BLOCKED_COMPONENT_TIME_LIMIT_NOT_MODEL_SCORE", timed_out=True
            )
            save()
            return result
        actual = json.loads(docker("inspect", identity))[0]
        result.update(
            container_exit_code=int(waited.stdout.strip()),
            oom_killed=actual["State"]["OOMKilled"],
        )
        with (
            (output / "stdout-private.log").open("xb") as stdout,
            (output / "stderr-private.log").open("xb") as stderr,
        ):
            subprocess.run(
                ["docker", "logs", identity],
                stdout=stdout,
                stderr=stderr,
                timeout=30,
                check=True,
            )
        if result["container_exit_code"] != 0 or result["oom_killed"]:
            result["status"] = "BLOCKED_WORKER_FAILURE_NOT_MODEL_SCORE"
            save()
            return result
        component_path = output / "component-result-private.json"
        export_file(
            holder,
            "/output/job/component-result-private.json",
            component_path,
            1024 * 1024,
        )
        component = json.loads(component_path.read_text(encoding="utf-8"))
        if (
            component.get("task_id") != args.task_id
            or component.get("parsed_code_sha256") != args.code_sha256
            or component.get("original_renderer_sha256") != RENDERERS[kind][0]
            or type(component.get("render_score")) is not int
            or component["render_score"] not in (0, 1)
        ):
            raise ValueError("returned component identity/result mismatch")
        export_file(
            holder,
            "/output/job/renderer-private.log",
            output / "renderer-private.log",
            64 * 1024**2,
        )
        image_present = component.get("image_present", bool(component.get("png_valid")))
        if image_present:
            png = output / (args.task_id + ".png")
            export_file(
                holder,
                "/output/job/" + args.task_id + ".png",
                png,
                64 * 1024**2,
            )
            with png.open("rb") as stream:
                if (
                    component.get("png_valid")
                    and stream.read(8) != b"\x89PNG\r\n\x1a\n"
                ):
                    raise ValueError("invalid exported PNG signature")
                stream.seek(0)
                result["png_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        result.update(
            status="ORIGINAL_RENDERER_COMPONENT_RETURNED_VQA_PENDING",
            render_score=component["render_score"],
            png_valid=bool(component.get("png_valid")),
            image_present=image_present,
            timed_out=False,
        )
        save()
        try:
            stop_owned(holder, operation, role="export")
            result["exporter_stopped_after_artifact_export"] = True
        except Exception as error:
            result["exporter_cleanup_pending_error_type"] = type(error).__name__
        save()
        return result
    except Exception as error:
        result.update(
            status="BLOCKED_COMPONENT_CONTROLLER_FAILURE",
            error_type=type(error).__name__,
        )
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--renderer", type=Path, required=True)
    parser.add_argument("--code-file", type=Path, required=True)
    parser.add_argument("--code-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--renderer-kind", choices=sorted(RENDERERS), default="matplotlib"
    )
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        parser.exit(
            1,
            f"StructEval controller failed ({type(error).__name__}); inspect private receipt\n",
        )
    print(json.dumps(result, indent=2))
    if result["status"].startswith("BLOCKED"):
        parser.exit(2)


if __name__ == "__main__":
    main()
