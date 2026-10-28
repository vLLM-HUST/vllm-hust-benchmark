"""Stage original tasks and score saved candidates in an offline local image.

Default prints a plan. No model calls, image pulls or host task execution.
Named-volume staging supports a Docker daemon in a separate container.
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

from livecodebench_container_worker import AUTHOR_SHA
from livecodebench_repair import repair_identities


def staging_directory(path):
    path.mkdir(mode=0o755)
    # The parent work root stays0700; only fresh copied sandbox inputs need
    # directory traversal by the nonroot worker even under operator umask077.
    path.chmod(0o755)
    return path


def worker_command(
    image,
    name,
    model,
    uid=65534,
    *,
    prepare_prompts=False,
    scenario="codegeneration",
    repair_feedback_compat="strict",
):
    if scenario not in {"codegeneration", "selfrepair"}:
        raise ValueError("explicit supported original scenario required")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("immutable existing local image ID required")
    if not re.fullmatch(r"hust-lcb-[0-9a-f]{32}", name) or not 1 <= uid <= 2**31 - 2:
        raise ValueError("owned container name and nonroot UID required")
    command = [
        "docker",
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
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=512m,mode=1777",
        "--mount",
        f"type=volume,source={name}-inputs,target=/inputs,readonly",
        "--mount",
        f"type=volume,source={name}-output,target=/output",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "HF_HOME=/tmp/huggingface",
        "--env",
        "OPENAI_KEY=unused-local-api",  # pragma: allowlist secret - fixed dummy local API key
        "--env",
        "OPENAI_BASE_URL=http://127.0.0.1:19466/v1",
        "--entrypoint",
        "python",
        image,
        "-B",
        "/inputs/worker.py",
        "--author-archive",
        "/inputs/source.tar.gz",
        "--tasks",
        "/inputs/tasks.json",
        "--generations",
        "/inputs/generations",
        "--output",
        "/output/results",
        "--model",
        model,
        "--scenario",
        scenario,
    ]
    if scenario == "selfrepair":
        if repair_feedback_compat not in {"strict", "error-message-only"}:
            raise ValueError(
                "explicit native or approved feedback compatibility required"
            )
        command += [
            "--prior-evaluation",
            "/inputs/prior-evaluation.json",
            "--repair-feedback-compat",
            repair_feedback_compat,
        ]
    return command + (["--prepare-prompts"] if prepare_prompts else [])


def execute(args):
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError(
            "authorized Linux root staging required; no host scoring fallback"
        )
    prepared = args.prepared_dir.resolve()
    if args.scenario == "selfrepair" and args.prior_evaluation is None:
        raise ValueError("selfrepair requires actual original evaluated feedback")
    generations = args.generations_dir.resolve() if args.generations_dir else None
    archive, work = args.author_archive.resolve(), args.work_root.resolve()
    if work.exists() or any(
        work.is_relative_to(p) or p.is_relative_to(work)
        for p in (prepared, generations)
        if p is not None
    ):
        raise ValueError("fresh work root independent from existing evidence required")
    if hashlib.sha256(archive.read_bytes()).hexdigest() != AUTHOR_SHA:
        raise ValueError("fixed author source archive mismatch")
    task_path = prepared / "codegen-original-tasks.private.json"
    original = json.loads((prepared / "receipt-private.json").read_text("utf-8"))
    if (
        hashlib.sha256(task_path.read_bytes()).hexdigest()
        != original["task_file_sha256"]
        or original["private_tests_decoded"] is not False
    ):
        raise ValueError("complete original generation and task relationships required")
    expected_repairs = []
    if args.scenario == "selfrepair":
        previous = json.loads(args.prior_evaluation.read_text("utf-8"))
        expected_repairs = repair_identities(previous)
        if [row["question_id"] for row in previous] != original[
            "selected_question_ids"
        ]:
            raise ValueError(
                "original evaluated tasks differ from selected original source"
            )
    need_generation = not args.prepare_prompts and (
        args.scenario == "codegeneration" or bool(expected_repairs)
    )
    if need_generation and generations is None:
        raise ValueError("saved original generations required for scoring")
    if need_generation:
        generated = json.loads(
            (generations / "receipt-private.json").read_text("utf-8")
        )
        valid = (
            generated["status"] == "ORIGINAL_GENERATION_RETURNED_NATIVE_SCORING_PENDING"
            and generated["completed_question_ids"] == original["selected_question_ids"]
        )
        if args.scenario == "selfrepair":
            valid = (
                generated["status"]
                == "ORIGINAL_SELF_REPAIR_GENERATION_RETURNED_NATIVE_SCORING_PENDING"
                and generated.get("completed_request_ids") == expected_repairs
            )
        if not valid:
            raise ValueError(
                "complete original generation and task relationships required"
            )
    name = "hust-lcb-" + uuid.uuid4().hex
    command = worker_command(
        args.image,
        name,
        args.model,
        prepare_prompts=args.prepare_prompts,
        scenario=args.scenario,
        repair_feedback_compat=args.repair_feedback_compat,
    )
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", args.image], text=True)
    )
    if len(image) != 1 or image[0]["Id"] != args.image:
        raise ValueError("local immutable image identity mismatch")
    work.mkdir(mode=0o700, parents=True)
    inputs = work / "inputs"
    staging_directory(inputs)
    staging_directory(inputs / "generations")
    for source, target in [
        (archive, inputs / "source.tar.gz"),
        (task_path, inputs / "tasks.json"),
        (
            Path(__file__).with_name("livecodebench_container_worker.py"),
            inputs / "worker.py",
        ),
        (
            Path(__file__).with_name("livecodebench_repair.py"),
            inputs / "livecodebench_repair.py",
        ),
    ]:
        shutil.copyfile(source, target)
        target.chmod(0o444)
    if args.scenario == "selfrepair":
        shutil.copyfile(args.prior_evaluation, inputs / "prior-evaluation.json")
        (inputs / "prior-evaluation.json").chmod(0o444)
    generation_count = (
        len(expected_repairs)
        if args.scenario == "selfrepair"
        else len(original["selected_question_ids"])
    )
    for index in range(0 if args.prepare_prompts else generation_count):
        filename = f"{index}.request-response.private.json"
        target = inputs / "generations" / filename
        shutil.copyfile(generations / filename, target)
        target.chmod(0o444)
    receipt = {
        "source_id": "LIVECODEBENCH",
        "status": "STAGING_OFFLINE_ORIGINAL_SCORING",
        "container_name": name,
        "image_id": args.image,
        "model_calls": 0,
        "source_checkbox_eligible": False,
        "other_native_scenarios_complete": False,
        "scenario": args.scenario,
        "repair_feedback_compat": args.repair_feedback_compat
        if args.scenario == "selfrepair"
        else None,
    }
    try:
        with (work / "container-private.log").open("xb") as log:

            def docker(*arguments):
                subprocess.run(
                    ["docker", *arguments],
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

            for key in ("inputs", "output"):
                docker(
                    "volume",
                    "create",
                    "--label",
                    "hust.workload.owner=" + name,
                    name + "-" + key,
                )
            docker(
                "create",
                "--name",
                name + "-stage",
                "--pull=never",
                "--network=none",
                "--mount",
                f"type=volume,source={name}-inputs,target=/inputs",
                "--mount",
                f"type=volume,source={name}-output,target=/output",
                "--entrypoint",
                "python",
                args.image,
                "-I",
                "-c",
                "pass",
            )
            docker("cp", str(inputs) + "/.", name + "-stage:/inputs")
            docker(
                "run",
                "--name",
                name + "-ownership",
                "--pull=never",
                "--network=none",
                "--read-only",
                "--user",
                "0:0",
                "--cap-drop=ALL",
                "--cap-add=CHOWN",
                "--cap-add=FOWNER",
                "--security-opt=no-new-privileges:true",
                "--memory",
                "256m",
                "--pids-limit",
                "32",
                "--mount",
                f"type=volume,source={name}-output,target=/output",
                "--entrypoint",
                "python",
                args.image,
                "-I",
                "-c",
                "import os; os.chown('/output',65534,65534); os.chmod('/output',0o700)",
            )
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            receipt["actual_exit_code"] = result.returncode
            docker("cp", name + ":/output/.", str(work / "native-output"))
        info = json.loads(
            subprocess.check_output(["docker", "inspect", name], text=True)
        )[0]
        receipt["container_actual_exit_code"] = info["State"]["ExitCode"]
        receipt["oom_killed"] = info["State"]["OOMKilled"]
        if (
            result.returncode != 0
            or info["State"]["ExitCode"] != 0
            or info["State"]["OOMKilled"]
        ):
            raise RuntimeError(
                "original scoring failure retained; do not rerun or change tasks blindly"
            )
        native = json.loads(
            (work / "native-output/results/receipt-private.json").read_text("utf-8")
        )
        if native["selected_question_ids"] != original["selected_question_ids"]:
            raise ValueError("original scoring question relationships incomplete")
        receipt["status"] = (
            "ORIGINAL_API_PROMPTS_PREPARED_NO_MODEL_CALLS"
            if args.prepare_prompts
            else "ORIGINAL_SELFREPAIR_SAMPLE_SCORED_OTHER_SCENARIOS_PENDING"
            if args.scenario == "selfrepair"
            else "ORIGINAL_CODEGEN_SAMPLE_SCORED_OTHER_SCENARIOS_PENDING"
        )
    except BaseException as error:
        receipt.update(
            status="STOP_RETAINED_NEEDS_TRIAGE", error_type=type(error).__name__
        )
        raise
    finally:
        (work / "receipt-private.json").write_text(
            json.dumps(receipt, indent=2), encoding="utf-8"
        )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--generations-dir", type=Path)
    parser.add_argument("--author-archive", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--prepare-prompts", action="store_true")
    parser.add_argument(
        "--scenario", choices=["codegeneration", "selfrepair"], default="codegeneration"
    )
    parser.add_argument("--prior-evaluation", type=Path)
    parser.add_argument(
        "--repair-feedback-compat",
        choices=["strict", "error-message-only"],
        default="strict",
    )
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "OFFLINE_PLAN_ONLY",
                    "model_calls": 0,
                    "source_checkbox_eligible": False,
                    "native_scoring_complete": False,
                }
            )
        )
        return
    print(json.dumps(execute(args), indent=2))


if __name__ == "__main__":
    main()
