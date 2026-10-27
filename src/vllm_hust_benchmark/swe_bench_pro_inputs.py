"""Pinned upstream V1 inputs, ARM images and independent official grading.

Migrated from request-throttling-controller at ea13820; upstream cores unchanged.
"""

import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

LOCK_PATH = Path(__file__).parent / "data/swe_bench_pro_v1_lock.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(configuration):
    from .swe_bench_pro import validate as check

    return check(configuration)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def command(args, output, timeout=900, cwd=None):
    start = time.monotonic()
    result = subprocess.run(
        args, capture_output=True, text=True, timeout=timeout, cwd=cwd
    )
    row = {
        "command": args,
        "exit_code": result.returncode,
        "elapsed_seconds": time.monotonic() - start,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "cwd": None if cwd is None else str(Path(cwd).resolve()),
    }
    save(output, row)
    if result.returncode:
        raise RuntimeError(f"Command failed; see {output}")
    return result.stdout


def validate_grading_sources(official, rows, lock):
    """Official V1 reads Dockerfiles relative to its process working directory."""
    official = Path(official)
    expected = (
        lock["official_v1"]["sha256"]
        | lock["official_v1"]["nodebb_task_source_sha256"]
        | lock["official_v1"].get("grading_dockerfiles_sha256", {})
    )
    checked = {}
    for row in rows:
        identifier = row["instance_id"]
        for relative in [
            f"dockerfiles/{kind}/{identifier}/Dockerfile"
            for kind in ["base_dockerfile", "instance_dockerfile"]
        ] + [
            f"run_scripts/{identifier}/{name}"
            for name in ["run_script.sh", "parser.py"]
        ]:
            if relative not in expected or not (official / relative).is_file():
                raise ValueError("Missing frozen official grading input: " + relative)
            if digest(official / relative) != expected[relative]:
                raise ValueError("Official grading source hash mismatch: " + relative)
            checked[relative] = expected[relative]
    for relative, value in lock["official_v1"]["sha256"].items():
        if digest(official / relative) != value:
            raise ValueError("Official evaluator source hash mismatch: " + relative)
        checked[relative] = value
    return checked


def prepare(configuration, output):
    """Materialize only selected private inputs, immutable images and source receipts."""
    import pyarrow.parquet as parquet

    c = validate(configuration)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lock = read(LOCK_PATH)
    paths = c["paths"]
    registry_args = ["--insecure"] if c.get("image_registry_insecure", False) else []
    assert digest(paths["dataset_parquet"]) == lock["dataset"]["parquet_sha256"]
    agent_source = Path(paths["agent_source"])
    for relative, expected in lock["agent_source_files"].items():
        assert digest(agent_source / relative) == expected, relative
    official = Path(paths["official_v1_source"])
    for relative, expected in lock["official_v1"]["sha256"].items():
        assert digest(official / relative) == expected, relative
    helper = runpy.run_path(str(official / "helper_code/image_uri.py"))
    rows = parquet.read_table(paths["dataset_parquet"]).to_pylist()
    selected = c["batch"].get("instance_ids")
    rows = sorted(
        (
            x
            for x in rows
            if x["repo"] == c["batch"]["repository"]
            and (not selected or x["instance_id"] in selected)
        ),
        key=lambda x: x["instance_id"],
    )[: c["batch"]["size"]]
    assert len(rows) == c["batch"]["size"]
    grading_sources = validate_grading_sources(official, rows, lock)
    os.environ["DOCKER_HOST"] = paths["docker_host"]
    case_bytes = "".join(
        json.dumps(x, ensure_ascii=False) + "\n" for x in rows
    ).encode()
    (output / "cases-private.jsonl").write_bytes(case_bytes)
    manifest = {
        "dataset": lock["dataset"],
        "cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
        "selection": c["batch"],
        "tasks": [],
        "official_sources": grading_sources,
    }
    save(output / "input-selection.json", manifest)
    for index, row in enumerate(rows):
        identifier = row["instance_id"]
        uri = helper["get_dockerhub_image_uri"](
            identifier, lock["community_images"]["dockerhub_username"], row["repo"]
        )
        mirror = c["image_mirror_prefix"] + uri
        raw = command(
            [
                paths["crane"],
                *registry_args,
                "manifest",
                "--platform=linux/arm64",
                mirror,
            ],
            output / f"image-{index}-manifest-command.json",
        )
        image_manifest = json.loads(raw)
        frozen_manifest = (
            "sha256:" + hashlib.sha256(raw.encode().rstrip(b"\n")).hexdigest()
        )
        # crane digest reads the registry's actual selected-platform digest.
        frozen_manifest = command(
            [
                paths["crane"],
                *registry_args,
                "digest",
                "--platform=linux/arm64",
                mirror,
            ],
            output / f"image-{index}-digest-command.json",
        ).strip()
        image_id = image_manifest["config"]["digest"]
        inspected = subprocess.run(
            [paths["docker"], "image", "inspect", image_id],
            capture_output=True,
            text=True,
        )
        if inspected.returncode:
            archive = output / f"image-{index}.tar"
            immutable = mirror.rsplit(":", 1)[0] + "@" + frozen_manifest
            command(
                [
                    paths["crane"],
                    *registry_args,
                    "pull",
                    "--platform=linux/arm64",
                    immutable,
                    str(archive),
                ],
                output / f"image-{index}-download.json",
                timeout=1200,
            )
            archive_hash = digest(archive)
            command(
                [paths["docker"], "load", "-i", str(archive)],
                output / f"image-{index}-load.json",
                timeout=300,
            )
            if not c["keep_image_archives"]:
                archive.unlink()  # Re-downloadable input only; measurements are never removed.
        else:
            archive_hash = None
        info = json.loads(
            command(
                [paths["docker"], "image", "inspect", image_id],
                output / f"image-{index}-inspect.json",
            )
        )[0]
        assert info["Architecture"] == "arm64" and info["Os"] == "linux"
        command(
            [paths["docker"], "tag", image_id, uri], output / f"image-{index}-tag.json"
        )
        identity = command(
            [
                paths["docker"],
                "run",
                "--rm",
                "--network=none",
                "--entrypoint=bash",
                "-w",
                "/app",
                image_id,
                "-lc",
                "uname -m && git rev-parse HEAD",
            ],
            output / f"image-{index}-base-check.json",
        )
        assert "aarch64" in identity and row["base_commit"] in identity
        for name in ("parser.py", "run_script.sh"):
            relative = f"run_scripts/{identifier}/{name}"
            actual = digest(official / relative)
            assert (
                actual == lock["official_v1"]["nodebb_task_source_sha256"][relative]
            ), relative
            manifest["official_sources"][relative] = actual
        task = {
            "instance_id": identifier,
            "base_commit": row["base_commit"],
            "image_uri": uri,
            "manifest_digest": frozen_manifest,
            "image_id": image_id,
            "download_archive_sha256": archive_hash,
        }
        manifest["tasks"].append(task)
        save(output / "input-selection.json", manifest)
        print("TASK_IMAGE_READY", json.dumps(task), flush=True)
    manifest["status"] = "PREPARED"
    save(output / "input-selection.json", manifest)
    return manifest


def grade_saved_batch(c, inputs, batch_directory, output, task_offset=0):
    """Grade saved patches only, without a model call or modifying old evidence."""
    from .swe_bench_pro_grading import summarize_application

    inputs, batch_directory, output = [
        Path(x).resolve() for x in (inputs, batch_directory, output)
    ]
    os.environ["DOCKER_HOST"] = c["paths"]["docker_host"]
    output.mkdir(parents=True, exist_ok=False)
    rows = [
        json.loads(x) for x in (inputs / "cases-private.jsonl").read_text().splitlines()
    ]
    copies = c["batch"].get("independent_agents_per_task", 1)
    if copies > 1:
        # Official scoring keys by instance_id. Separate invocations prevent
        # independent patches for the same task from overwriting each other.
        single = json.loads(json.dumps(c))
        single["batch"]["independent_agents_per_task"] = 1
        scores, unresolved, application = {}, [], {}
        for replica in range(copies):
            label = f"replica-{replica:03d}"
            graded = grade_saved_batch(
                single,
                inputs,
                batch_directory,
                output / label,
                task_offset=replica * len(rows),
            )
            scores.update(
                {label + ":" + k: v for k, v in graded["official_scores"].items()}
            )
            unresolved.extend(label + ":" + k for k in graded["unresolved_grading"])
            application.update(
                {label + ":" + k: v for k, v in graded["patch_application"].items()}
            )
        result = {
            "official_scores": scores,
            "patch_application": application,
            "unresolved_grading": unresolved,
            "grading_status": "OUTPUTS_PRESENT"
            if not unresolved
            else "UNRESOLVED_OUTPUTS",
            "unique_official_tasks": len(rows),
            "independent_agent_instances": len(rows) * copies,
            "model_calls_in_this_grading": 0,
        }
        save(output / "grading-status.json", result)
        return result
    lock = read(LOCK_PATH)
    official = Path(c["paths"]["official_v1_source"])
    sources = validate_grading_sources(official, rows, lock)
    adapter = Path(__file__).with_name("swe_bench_pro_grading.py")
    helper = Path(__file__).with_name("swe_patch_state.py")
    patches, bindings = [], []
    for index, row in enumerate(rows):
        patch = batch_directory / f"task-{index + task_offset:03d}/patch-private.diff"
        patches.append(
            {
                "instance_id": row["instance_id"],
                "patch": patch.read_text(),
                "prefix": "agent",
            }
        )
        bindings.append(
            {"instance_id": row["instance_id"], "patch_sha256": digest(patch)}
        )
    save(
        output / "source-and-patch-bindings.json",
        {
            "official_sources": sources,
            "patches": bindings,
            "application_evidence_sources": {
                path.name: digest(path) for path in (adapter, helper)
            },
        },
    )
    save(output / "patches-private.json", patches)
    task_images = read(inputs / "input-selection.json")["tasks"]

    def check_images(label):
        images = {}
        for task in task_images:
            raw = command(
                [c["paths"]["docker"], "image", "inspect", task["image_uri"]],
                output / f"{label}-{len(images)}-image-private.json",
            )
            image = json.loads(raw)[0]
            if image["Id"] != task["image_id"]:
                raise ValueError(
                    "Grading image changed from frozen Agent input: "
                    + task["instance_id"]
                )
            images[task["instance_id"]] = image["Id"]
        save(output / (label + "-image-ids.json"), images)

    check_images("before")
    arguments = [
        sys.executable,
        str(adapter),
        "--official-script",
        str(official / "swe_bench_pro_eval.py"),
        "--official-sha256",
        digest(official / "swe_bench_pro_eval.py"),
        "--raw_sample_path",
        str(inputs / "cases-private.jsonl"),
        "--patch_path",
        str(output / "patches-private.json"),
        "--output_dir",
        str(output),
        "--scripts_dir",
        str(official / "run_scripts"),
        "--dockerhub_username",
        "bdqnghi",
        "--use_local_docker",
        "--docker_platform",
        "linux/arm64",
        "--num_workers",
        "1",
    ]
    with (output / "docker-events-private.jsonl").open("w") as events:
        observer = subprocess.Popen(
            [
                c["paths"]["docker"],
                "events",
                "--since",
                str(int(time.time())),
                "--filter",
                "type=container",
                "--format",
                "{{json .}}",
            ],
            stdout=events,
            stderr=subprocess.STDOUT,
        )
        try:
            command(
                arguments, output / "command-private.json", timeout=2400, cwd=official
            )
        finally:
            observer.terminate()
            observer.wait(timeout=10)
    check_images("after")
    scores = read(output / "eval_results.json")
    unresolved = [
        row["instance_id"]
        for row in rows
        if not (output / row["instance_id"] / "workspace/output.json").is_file()
    ]
    result = {
        "official_scores": scores,
        "patch_application": {
            row["instance_id"]: summarize_application(
                output / row["instance_id"] / "workspace"
            )
            for row in rows
        },
        "unresolved_grading": unresolved,
        "grading_status": "OUTPUTS_PRESENT" if not unresolved else "UNRESOLVED_OUTPUTS",
        "model_calls_in_this_grading": 0,
    }
    save(output / "grading-status.json", result)
    print("SAVED_PATCH_GRADING_FINISHED", json.dumps(result), flush=True)
    return result
