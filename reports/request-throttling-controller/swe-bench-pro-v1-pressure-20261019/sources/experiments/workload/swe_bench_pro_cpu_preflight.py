"""Model-free resource/identity preflight using real independent ARM containers.

Use as a named Ray Job in the owned CPU workspace. This is preparation evidence,
not Agent inference, task scoring, or an engine-performance measurement.
"""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time
import traceback

from run_swe_bench_pro import digest, read, save, validate
from swe_bench_pro_batch import expand_agent_instances, program_identifier


def child(c, row, image, folder, ready, release):
    folder.mkdir(exist_ok=False)
    env = None
    try:
        os.environ.update(DOCKER_HOST=c["paths"]["docker_host"], MSWEA_SILENT_STARTUP="1",
                          LITELLM_LOCAL_MODEL_COST_MAP="True")
        sys.path.insert(0, str(Path(c["paths"]["agent_source"]) / "src"))
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments.docker import DockerEnvironment
        from minisweagent.models.litellm_model import LitellmModel
        assert DefaultAgent and LitellmModel
        env = DockerEnvironment(image=image["image_id"], cwd="/app", executable=c["paths"]["docker"],
                                timeout=c["agent"]["tool_timeout_seconds"], container_timeout="10m",
                                run_args=["--rm", "--entrypoint", "", "--network", c["container"]["network"],
                                          "--cpus", str(c["container"]["cpus"]), "--memory", c["container"]["memory"]])
        (folder / "container-id.txt").write_text(env.container_id)
        result = env.execute("git rev-parse HEAD && node --version && redis-server --version")
        if result["returncode"] != 0 or result["output"].splitlines()[0] != row["base_commit"]:
            raise ValueError("Real task code/runtime mismatch")
        inspected = json.loads(subprocess.check_output([c["paths"]["docker"], "inspect", env.container_id]))[0]
        if inspected["Image"] != image["image_id"] or inspected["Mounts"]:
            raise ValueError("Immutable image or independent container boundary mismatch")
        record = {"ready": True, "pid": os.getpid(), "instance_id": row["instance_id"],
                  "program_id": program_identifier(c, row, "cpu-preflight"),
                  "container_id": env.container_id, "image_id": inspected["Image"],
                  "code_and_runtime": result, "loaded_agent_file": sys.modules["minisweagent"].__file__}
        save(folder / "ready-private.json", record)
        ready.put(record)
        if not release.wait(240):
            raise TimeoutError("Parent did not release CPU preflight")
    except Exception:
        save(folder / "failure-private.json", {"traceback": traceback.format_exc()})
        ready.put({"ready": False, "folder": str(folder), "error": traceback.format_exc()})
    finally:
        if env and env.container_id:
            subprocess.run([c["paths"]["docker"], "rm", "-f", env.container_id], capture_output=True)
            env.container_id = None


def run(c, inputs, output):
    c = validate(c)
    inputs, output = Path(inputs), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["DOCKER_HOST"] = c["paths"]["docker_host"]
    manifest = read(inputs / "input-selection.json")
    if digest(inputs / "cases-private.jsonl") != manifest["cases_sha256"]:
        raise ValueError("Frozen input bytes changed")
    rows = [json.loads(x) for x in (inputs / "cases-private.jsonl").read_text().splitlines()]
    rows, images = expand_agent_instances(c, rows, manifest["tasks"])
    if shutil.disk_usage(output).free < 4 * 2**30:
        raise ValueError("Insufficient private volume space for preflight")
    context = multiprocessing.get_context("spawn")
    ready, release = context.Queue(), context.Event()
    processes, observations, records = [], [], []
    start = time.monotonic()
    try:
        for i, (row, image) in enumerate(zip(rows, images)):
            p = context.Process(target=child, args=(c, row, image, output / f"task-{i:03d}", ready, release))
            p.start()
            processes.append(p)
        deadline = start + 180
        while len(records) < len(rows):
            current = Path("/sys/fs/cgroup/memory.current")
            used = int(current.read_text()) if current.exists() else None
            observations.append({"elapsed_seconds": time.monotonic() - start, "dev_cgroup_bytes": used})
            if used is not None and used > 48 * 2**30:
                raise MemoryError("Dev preflight exceeded 48 GiB guard within 64 GiB budget")
            if time.monotonic() > deadline:
                raise TimeoutError("Not all containers/Agent imports ready within original 180s barrier budget")
            try:
                record = ready.get(timeout=1)
            except queue.Empty:
                continue
            records.append(record)
            if not record["ready"]:
                raise RuntimeError("Real CPU preflight child failed; see private logs")
            print("CPU_PREFLIGHT_READY", len(records), "/", len(rows), flush=True)
        stats = subprocess.check_output([c["paths"]["docker"], "stats", "--no-stream", "--format", "{{json .}}"], text=True)
        (output / "docker-stats-private.jsonl").write_text(stats)
        if len({x["container_id"] for x in records}) != len(rows) or len({x["program_id"] for x in records}) != len(rows):
            raise ValueError("Independent container/program identities collided")
        result = {"status": "ALL_INDEPENDENT_CONTAINERS_AND_AGENT_IMPORTS_READY", "model_calls": 0,
                  "program_instances": len(rows), "unique_official_tasks": len({x["instance_id"] for x in rows}),
                  "elapsed_seconds": time.monotonic() - start, "dev_memory_observations": observations,
                  "remaining_disk_bytes": shutil.disk_usage(output).free,
                  "boundary": "Agent imports and container/code/runtime readiness only; tool execution peaks and model pressure remain untested"}
        save(output / "report.json", result)
        print("CPU_PREFLIGHT_PASS", len(rows), flush=True)
    except Exception:
        save(output / "failure-private.json", {"traceback": traceback.format_exc(), "ready_instances": len(records),
                                               "observations": observations, "model_calls": 0})
        raise
    finally:
        release.set()
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        cleanup = []
        for path in output.glob("task-*/container-id.txt"):
            v = subprocess.run([c["paths"]["docker"], "rm", "-f", path.read_text()], capture_output=True, text=True)
            cleanup.append({"container_id": path.read_text(), "returncode": v.returncode, "stderr": v.stderr})
        save(output / "cleanup-private.json", cleanup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(read(args.settings), args.inputs, args.output)
