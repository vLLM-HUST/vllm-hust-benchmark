from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


HOST_COMMIT = "973d9ef5acd2b9beb03e399dec1511a41fc941b5"
ASCEND_COMMIT = "873584026f8a683ac73523f15127569e9f6ab34b"
PROFILER_COMMIT = "88ffdce5fc6639c0b7b86a54d197f7394705c54b"
KVCOMPRESS_COMMIT = "00cc78dc7f3491eecbc79c5068e29aed9f59afc3"
MODEL = Path("/data/shared_models/Qwen--Qwen2.5-14B-Instruct")
STATS = Path(
    "/root/ecpa-h2-real-pair-20261013/calibration-current-mount-v2/"
    "qwen2.5-14b-current-mount-stats-v3.pt"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def npu_info(out: Path, name: str) -> str:
    completed = subprocess.run(
        ["npu-smi", "info"], check=True, capture_output=True, text=True
    )
    (out / name).write_text(completed.stdout)
    return completed.stdout


def identity(pid: int) -> dict[str, object]:
    stat = Path(f"/proc/{pid}/stat").read_text()
    tail = stat[stat.rfind(")") + 2 :].split()
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    return {
        "pid": pid,
        "start_ticks": int(tail[19]),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "argv": [x.decode(errors="surrogateescape") for x in cmdline if x],
    }


def capture_tree(pid: int, values: dict[int, dict[str, object]]) -> None:
    try:
        values[pid] = identity(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        for child in children:
            capture_tree(int(child), values)
    except (OSError, ValueError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--order", required=True)
    parser.add_argument("--port", type=int, default=8197)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(mode=0o700, parents=True, exist_ok=False)

    campaign = json.loads(args.campaign.read_text())
    oracle = json.loads(args.oracle.read_text())
    if campaign.get("schema") != "ecpa-h2-compression-transaction-campaign/v1":
        raise RuntimeError("unsupported campaign schema")
    if campaign.get("formal_runs") != []:
        raise RuntimeError("campaign was not frozen before execution")
    if campaign.get("runner", {}).get("sha256") != sha256(Path(__file__)):
        raise RuntimeError("campaign runner digest mismatch")
    if campaign.get("oracle_sha256") != "sha256:" + sha256(args.oracle):
        raise RuntimeError("campaign oracle digest mismatch")
    if oracle.get("schema") != "ecpa-h2-compression-transaction-oracle/v1":
        raise RuntimeError("unsupported oracle schema")
    if oracle.get("planner_runtime_blind") is not True:
        raise RuntimeError("oracle is not declared planner/runtime blind")
    if oracle.get("label") != "compatible-compression-transaction-expected":
        raise RuntimeError("oracle does not qualify this transaction")

    frozen_members = {item["id"]: item for item in campaign["members"]}
    for member_id, commit in {
        "lifecycle_profiler": PROFILER_COMMIT,
        "kvcompress": KVCOMPRESS_COMMIT,
    }.items():
        if frozen_members.get(member_id, {}).get("commit") != commit:
            raise RuntimeError(f"campaign source mismatch: {member_id}")
    host = campaign["host"]
    if (
        host["core"]["commit"] != HOST_COMMIT
        or host["ascend"]["commit"] != ASCEND_COMMIT
        or host["tensor_parallel_size"] != 2
    ):
        raise RuntimeError("campaign host identity mismatch")
    orders = {item["order_id"] for item in campaign["installation_orders"]}
    if args.order not in orders:
        raise RuntimeError("installation order is not frozen")

    workload = campaign["workload"]
    prompt = str(workload["prompt_unit"]) * int(workload["prompt_repeat"])
    prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    if "sha256:" + prompt_sha != workload["prompt_sha256"]:
        raise RuntimeError("generated prompt digest mismatch")
    tokenizer_script = (
        "from transformers import AutoTokenizer;"
        "import pathlib,sys;"
        f"t=AutoTokenizer.from_pretrained({str(MODEL)!r},local_files_only=True);"
        "print(len(t.encode(pathlib.Path(sys.argv[1]).read_text())))"
    )
    prompt_path = args.out / "prompt.txt"
    prompt_path.write_text(prompt)
    prompt_tokens = int(
        subprocess.check_output(
            [str(args.venv / "bin/python"), "-c", tokenizer_script, str(prompt_path)],
            text=True,
        ).strip().splitlines()[-1]
    )
    if prompt_tokens != int(workload["expected_prompt_tokens"]):
        raise RuntimeError("frozen prompt token count mismatch")

    method_config = dict(campaign["kvcompress"]["method_config"])
    method_config["stats_path"] = str(STATS)
    config = {
        "schema_version": 1,
        "provider": "ascend_kvcompress",
        "method": "triattention",
        "method_config": method_config,
    }
    threshold = int(method_config["kv_budget"]) + int(method_config["recompute_window"])
    if prompt_tokens < threshold:
        raise RuntimeError("frozen workload does not cross compression threshold")
    config_path = args.out / "kvcompress-config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    trace_path = args.out / "profiler-trace.jsonl"

    served = campaign["model"]["served_name_template"].format(order_id=args.order)
    argv = [
        str(args.venv / "bin/vllm-request-lifecycle-profiler-serve"),
        "serve", str(MODEL), "--served-model-name", served,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--tensor-parallel-size", "2", "--enforce-eager",
        "--max-model-len", str(campaign["model"]["max_model_len"]),
        "--gpu-memory-utilization", "0.7", "--block-size", "128",
        "--no-enable-prefix-caching",
    ]
    env = dict(os.environ)
    env.update({
        "PYTHONNOUSERSITE": "1",
        "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS": "12",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "OMP_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_PLUGINS": "ascend,ascend_kvcompress,request_lifecycle_profiler",
        "VLLM_ASCEND_KVCOMPRESS_ENABLED": "1",
        "VLLM_ASCEND_KVCOMPRESS_CONFIG": str(config_path),
        "VLLM_RLP_TRACE_EXPORT_PATH": str(trace_path),
        "VLLM_RLP_PROFILER_PARENT_COMMIT": PROFILER_COMMIT,
        "VLLM_RLP_RUNTIME_CORE_COMMIT": HOST_COMMIT,
        "VLLM_RLP_DEVICE_PLUGIN_COMMIT": ASCEND_COMMIT,
        "VLLM_RLP_COMMUNICATION_MODE": "none",
    })
    for name in (
        "VLLMHUST_EXT_ENABLED_BUNDLES", "ASCEND_RT_VISIBLE_DEVICES",
        "ECPA_ACTIVATION_CONTRACT", "VLLM_ECPA_PLAN_ID", "VLLM_ECPA_LAUNCH_ID",
        "VLLM_ECPA_EVIDENCE_SINK", "VLLM_ECPA_EVIDENCE_STRICT",
        "ECPA_HOST_EVENT_DIR", "ECPA_HOST_EVENT_FSYNC", "ECPA_HOST_EVENT_DEVICE",
        "ECPA_HOST_EVENT_INODE", "ECPA_CONTROLLER_INSTANCE",
    ):
        env.pop(name, None)

    package_script = (
        "import importlib.metadata as m,json;print(json.dumps({x:m.version(x) for x in "
        "('vllm','vllm-ascend','vllm-request-lifecycle-profiler-plugin',"
        "'vllm-ascend-kvcompress-hust')},sort_keys=True))"
    )
    result: dict[str, object] = {
        "schema": "ecpa-h2-compression-transaction-raw-run/v1",
        "classification": "formal-candidate-real-compression-transaction",
        "score_eligible": False,
        "performance_result": False,
        "order": args.order,
        "served_model": served,
        "runner_sha256": sha256(Path(__file__)),
        "campaign_sha256": sha256(args.campaign),
        "oracle_sha256": sha256(args.oracle),
        "prompt_sha256": prompt_sha,
        "prompt_tokens": prompt_tokens,
        "compression_threshold_tokens": threshold,
        "packages": json.loads(subprocess.check_output(
            [str(args.venv / "bin/python"), "-c", package_script], text=True
        )),
        "stats_sha256": sha256(STATS),
        "config_sha256": sha256(config_path),
        "argv": argv,
        "failure": None,
    }
    identities: dict[int, dict[str, object]] = {}
    process = None
    before = npu_info(args.out, "npu-before.txt")
    if before.count("No running processes found") != 2:
        raise RuntimeError("allocated NPU process tables are not empty")
    if port_open(args.port):
        raise RuntimeError("port is occupied")
    try:
        with (args.out / "stdout.bin").open("wb") as stdout, (args.out / "stderr.bin").open("wb") as stderr:
            process = subprocess.Popen(
                argv, env=env, cwd=args.out, stdout=stdout, stderr=stderr,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            result["server_identity"] = identity(process.pid)
            deadline = time.monotonic() + 720
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            while time.monotonic() < deadline:
                capture_tree(process.pid, identities)
                if process.poll() is not None:
                    raise RuntimeError(f"service exited before health: {process.returncode}")
                try:
                    with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                        if response.status == 200:
                            result["health_status"] = 200
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.5)
            else:
                raise RuntimeError("service readiness deadline exceeded")
            body = json.dumps({
                "model": served, "prompt": prompt,
                "max_tokens": int(workload["max_tokens"]),
                "temperature": workload["temperature"], "seed": int(workload["seed"]),
            }, sort_keys=True, separators=(",", ":")).encode()
            (args.out / "request.json").write_bytes(body + b"\n")
            request = urllib.request.Request(
                f"http://127.0.0.1:{args.port}/v1/completions", data=body,
                headers={"Content-Type": "application/json"},
            )
            with opener.open(request, timeout=300) as response:
                raw = response.read(1024 * 1024)
                result["completion_status"] = response.status
            (args.out / "response.json").write_bytes(raw)
            payload = json.loads(raw)
            result["response_id"] = payload.get("id")
            result["completion_nonempty"] = bool(payload["choices"][0]["text"].strip())
            result["completion_text"] = payload["choices"][0]["text"]
            result["usage"] = payload.get("usage")
            if not result["completion_nonempty"]:
                raise RuntimeError("completion was empty")
            time.sleep(3)
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
    finally:
        if process is not None:
            capture_tree(process.pid, identities)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=15)
            result["server_exit_code"] = process.returncode
        time.sleep(5)
        after = npu_info(args.out, "npu-after.txt")
        result["allocated_npu_process_tables_empty"] = after.count("No running processes found") == 2
        result["port_refused_after_stop"] = not port_open(args.port)
        result["process_identities"] = list(identities.values())
        result["surviving_owned_processes"] = [
            value for pid, value in identities.items() if Path(f"/proc/{pid}").exists()
        ]
        trace_records = []
        traces = sorted(args.out.glob("profiler-trace*.jsonl"))
        for path in traces:
            for line in path.read_text().splitlines():
                if line.strip():
                    trace_records.append(json.loads(line))
        result["profiler_trace_files"] = [path.name for path in traces]
        result["profiler_trace_records"] = trace_records
        logs = (args.out / "stderr.bin").read_text(errors="replace") + "\n" + (args.out / "stdout.bin").read_text(errors="replace")
        commits = re.findall(
            r"KV compression scheduler commit request_id=(\S+) semantic_tokens=(\d+) physical_tokens=(\d+) source_blocks=(\d+) destination_blocks=(\d+) released_blocks=(\d+)", logs
        )
        acks = re.findall(
            r"KV compression worker commit acknowledged request_id=(\S+) semantic_tokens=(\d+) physical_tokens=(\d+) destination_blocks=(\d+)", logs
        )
        result["scheduler_commits"] = commits
        result["worker_acknowledgements"] = acks
        matching = [
            {"request_id": c[0], "semantic_tokens": int(c[1]), "physical_tokens": int(c[2]), "destination_blocks": int(c[4])}
            for c in commits for a in acks
            if c[0] == a[0] and c[1] == a[1] and c[2] == a[2] and c[4] == a[3]
        ]
        result["matching_transactions"] = matching
        result["cleanup_proven"] = bool(
            result["allocated_npu_process_tables_empty"] and result["port_refused_after_stop"] and not result["surviving_owned_processes"]
        )
        result["success"] = bool(
            result["failure"] is None and result.get("health_status") == 200
            and result.get("completion_status") == 200 and result.get("completion_nonempty") is True
            and matching and result["cleanup_proven"]
            and any(row.get("event_name") == "request_finished_observed" for row in trace_records)
        )
        (args.out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        hashes = {
            str(path.relative_to(args.out)): sha256(path)
            for path in sorted(args.out.rglob("*")) if path.is_file() and path.name != "sha256.json"
        }
        (args.out / "sha256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
