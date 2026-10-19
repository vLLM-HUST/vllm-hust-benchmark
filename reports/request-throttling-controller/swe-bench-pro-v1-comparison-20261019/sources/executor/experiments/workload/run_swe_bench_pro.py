"""Reproducible V1 real-Agent batch entry; upstream owns Agent/scoring/policy.

Run on an existing Linux ARM Coder. No workspace, template or device creation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import time
import urllib.request

REPOSITORY = Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def http(url, data=None, timeout=30):
    request = urllib.request.Request(
        url, data=None if data is None else json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode()


def validate(configuration):
    if configuration["schema"] != "swe-bench-pro-experiment-v1":
        raise ValueError("Unsupported configuration schema")
    batch = configuration["batch"]
    if batch["arrival_time"] != 0 or batch["client_concurrency_limit"] is not None:
        raise ValueError("Finite batch must arrive at time zero without client slots")
    if type(batch["size"]) is not int or batch["size"] < 1:
        raise ValueError("Explicit positive batch size required")
    if configuration["protocol"]["pair_order"] != [
        ["native", "controller"], ["controller", "native"], ["native", "controller"]
    ]:
        raise ValueError("Frozen three-pair alternating protocol required")
    if configuration["agent"]["automatic_retries"] != 0:
        raise ValueError("No automatic model retry in this experiment")
    if not configuration["protocol"]["reset_prefix_cache_before_each_batch"]:
        raise ValueError("Identical empty-prefix-cache initial state required")
    for name in ("dataset_parquet", "agent_source", "official_v1_source", "docker"):
        if not Path(configuration["paths"][name]).is_absolute():
            raise ValueError("Deployment paths must be explicit and absolute: " + name)
    return configuration


def validate_workspace_lease(configuration, now=None):
    """Background Ray tasks do not by themselves extend a Coder deadline."""
    workspace = configuration.get("workspace", {})
    if workspace.get("no_deadline_confirmed") is True and workspace.get("deadline") is None:
        return {"status": "NO_DEADLINE_CONFIRMED"}
    value = workspace.get("deadline")
    if not value:
        raise ValueError("Capture the actual Coder deadline before submitting a suite")
    deadline = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if deadline.tzinfo is None:
        raise ValueError("Coder deadline must include its timezone")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    stages = sum(len(pair) for pair in configuration["protocol"]["pair_order"])
    required = stages * (configuration["agent"]["wall_budget_seconds"] + 2400) + 600
    remaining = (deadline - now).total_seconds()
    if remaining < required:
        raise ValueError("Coder lease does not cover frozen Agent/grader bounds and cleanup")
    return {"status": "LEASE_COVERS_FROZEN_BOUNDS", "remaining_seconds": remaining,
            "required_seconds": required, "deadline": value}


def runtime_shim(configuration, settings_path):
    deployment = configuration["model_deployment"]
    command = [deployment.get("bootstrap_python", "/usr/local/python3.12.13/bin/python3.12"),
               str(Path(configuration["paths"]["repository"]) /
                   "experiments/workload/swe_bench_pro_runtime.py"),
               "--settings", str(Path(settings_path).absolute()), "--"]
    return '#!/bin/bash\nexec ' + shlex.join(command) + ' "$@"\n'


def validate_deployment(plan, configuration):
    """A same-container TP2 serve Job must reserve its own Ray NPUs."""
    models = configuration["model_services"]
    if len(models) != 1 or models[0]["id"] != plan["model_service"]:
        raise ValueError("Exactly one named owned model service required")
    if models[0].get("resources", {}).get("NPU") != 2:
        raise ValueError("The TP2 multiprocess model Job must reserve NPU: 2 in Ray")
    proxies = configuration["controllers"]
    if len(proxies) != 2 or sorted(x["scheduling_enabled"] for x in proxies) != [False, True]:
        raise ValueError("Matched Native and Controller proxy pair required")
    if any(x["backend"] != models[0]["endpoint"] for x in proxies):
        raise ValueError("Both arms must use the same model service")
    if len({x["transport"] for x in proxies}) != 1:
        raise ValueError("Proxy transports must match")
    return {"model_job_npu_reservation": 2, "matched_proxies": 2}


def command(args, output, timeout=900):
    start = time.monotonic()
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    row = {"command": args, "exit_code": result.returncode,
           "elapsed_seconds": time.monotonic() - start,
           "stdout": result.stdout, "stderr": result.stderr}
    save(output, row)
    if result.returncode:
        raise RuntimeError(f"Command failed; see {output}")
    return result.stdout


def prepare(configuration, output):
    """Materialize only selected private inputs, immutable images and source receipts."""
    import pyarrow.parquet as parquet

    c = validate(configuration)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    lock = read(REPOSITORY / "config/swe-bench-pro-v1-lock.json")
    paths = c["paths"]
    assert digest(paths["dataset_parquet"]) == lock["dataset"]["parquet_sha256"]
    agent_source = Path(paths["agent_source"])
    for relative, expected in lock["agent_source_files"].items():
        assert digest(agent_source / relative) == expected, relative
    official = Path(paths["official_v1_source"])
    for relative, expected in lock["official_v1"]["sha256"].items():
        assert digest(official / relative) == expected, relative
    helper = runpy.run_path(str(official / "helper_code/image_uri.py"))
    rows = parquet.read_table(paths["dataset_parquet"]).to_pylist()
    rows = sorted((x for x in rows if x["repo"] == c["batch"]["repository"]),
                  key=lambda x: x["instance_id"])[:c["batch"]["size"]]
    assert len(rows) == c["batch"]["size"]
    os.environ["DOCKER_HOST"] = paths["docker_host"]
    case_bytes = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows).encode()
    (output / "cases-private.jsonl").write_bytes(case_bytes)
    manifest = {"dataset": lock["dataset"], "cases_sha256": hashlib.sha256(case_bytes).hexdigest(),
                "selection": c["batch"], "tasks": [], "official_sources": {}}
    save(output / "input-selection.json", manifest)
    for index, row in enumerate(rows):
        identifier = row["instance_id"]
        uri = helper["get_dockerhub_image_uri"](
            identifier, lock["community_images"]["dockerhub_username"], row["repo"])
        mirror = c["image_mirror_prefix"] + uri
        raw = command([paths["crane"], "manifest", "--platform=linux/arm64", mirror],
                      output / f"image-{index}-manifest-command.json")
        image_manifest = json.loads(raw)
        frozen_manifest = "sha256:" + hashlib.sha256(raw.encode().rstrip(b"\n")).hexdigest()
        # crane digest reads the registry's actual selected-platform digest.
        frozen_manifest = command([paths["crane"], "digest", "--platform=linux/arm64", mirror],
                                  output / f"image-{index}-digest-command.json").strip()
        image_id = image_manifest["config"]["digest"]
        inspected = subprocess.run([paths["docker"], "image", "inspect", image_id],
                                   capture_output=True, text=True)
        if inspected.returncode:
            archive = output / f"image-{index}.tar"
            immutable = mirror.rsplit(":", 1)[0] + "@" + frozen_manifest
            command([paths["crane"], "pull", "--platform=linux/arm64", immutable, str(archive)],
                    output / f"image-{index}-download.json", timeout=1200)
            archive_hash = digest(archive)
            command([paths["docker"], "load", "-i", str(archive)],
                    output / f"image-{index}-load.json", timeout=300)
            if not c["keep_image_archives"]:
                archive.unlink()  # Re-downloadable input only; measurements are never removed.
        else:
            archive_hash = None
        info = json.loads(command([paths["docker"], "image", "inspect", image_id],
                                  output / f"image-{index}-inspect.json"))[0]
        assert info["Architecture"] == "arm64" and info["Os"] == "linux"
        command([paths["docker"], "tag", image_id, uri], output / f"image-{index}-tag.json")
        identity = command([paths["docker"], "run", "--rm", "--network=none",
                            "--entrypoint=bash", "-w", "/app", image_id,
                            "-lc", "uname -m && git rev-parse HEAD"],
                           output / f"image-{index}-base-check.json")
        assert "aarch64" in identity and row["base_commit"] in identity
        for name in ("parser.py", "run_script.sh"):
            relative = f"run_scripts/{identifier}/{name}"
            actual = digest(official / relative)
            assert actual == lock["official_v1"]["nodebb_task_source_sha256"][relative], relative
            manifest["official_sources"][relative] = actual
        task = {"instance_id": identifier, "base_commit": row["base_commit"],
                "image_uri": uri, "manifest_digest": frozen_manifest,
                "image_id": image_id, "download_archive_sha256": archive_hash}
        manifest["tasks"].append(task)
        save(output / "input-selection.json", manifest)
        print("TASK_IMAGE_READY", json.dumps(task), flush=True)
    manifest["status"] = "PREPARED"
    save(output / "input-selection.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "shim", "prepare", "deploy", "run", "suite", "summarize"])
    parser.add_argument("--settings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--arm", choices=["native", "controller"])
    args = parser.parse_args()
    c = validate(read(args.settings))
    if args.action == "check":
        print(json.dumps({"configuration": "PASS", "batch_size": c["batch"]["size"]}))
    elif args.action == "shim":
        args.output.mkdir(parents=True, exist_ok=False)
        shim = args.output / "vllm"
        shim.write_text(runtime_shim(c, args.settings))
        shim.chmod(0o755)
        print(json.dumps({"shim": str(shim), "sha256": digest(shim)}))
    elif args.action == "prepare":
        prepare(c, args.output)
    elif args.action == "deploy":
        plan = read(c["deployment"]["ride_plan"])
        validate_deployment(plan, read(plan["config"]))
        subprocess.run(["bash", c["deployment"]["ride_one_click"], "deploy",
                        c["deployment"]["ride_plan"], str(args.output)], check=True)
    else:
        from swe_bench_pro_batch import run_batch, run_suite, summarize
        if args.action == "run":
            assert args.arm and args.inputs
            run_batch(c, args.inputs, args.arm, args.output)
        elif args.action == "suite":
            assert args.inputs
            run_suite(c, args.inputs, args.output)
        else:
            summarize(args.output)


if __name__ == "__main__":
    main()
