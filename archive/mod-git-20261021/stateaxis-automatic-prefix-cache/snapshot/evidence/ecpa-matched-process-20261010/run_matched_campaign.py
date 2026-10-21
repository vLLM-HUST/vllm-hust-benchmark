#!/usr/bin/env python3
"""Matched ECPA process campaign for the automatic-prefix-cache MOD.

This exercises the real StateAxis HTTP service, ECPA injection, Rust host
integration, and Native protocol lifecycle against the deterministic protocol
worker. It is explicitly not accelerator or performance evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATEAXIS = Path(os.environ.get("STATEAXIS_REPO", "/root/stateaxis"))
MOD_REPO = Path(
    os.environ.get(
        "APC_MOD_REPO",
        "/root/stateaxis-topic-mods/repositories/stateaxis-automatic-prefix-cache",
    )
)
MANAGER_REPO = Path(
    os.environ.get("EXTENSION_MANAGER_REPO", "/root/mod-modernization/manager-main")
)
CLI = Path(
    os.environ.get("VLLM_HUST_EXT", str(MANAGER_REPO / ".venv/bin/vllm-hust-ext"))
)
ENGINE = Path(
    os.environ.get(
        "STATEAXIS_NATIVE_ENGINE",
        str(STATEAXIS / "target/release/native_state_engine"),
    )
)
CONFIGURE = ROOT / "manager-config.json"
ARTIFACT = STATEAXIS / ".benchmarks/issue57-preemption-artifact-20260815-r2"
EXECUTION_CONFIG_SHA256 = (
    "09682f76a33b63dd5312aa7a0f7b43f9b2e1144cbd22c14ba952b9a9907e0815"
)
EXECUTION_ARTIFACT_SHA256 = (
    "f82b25e00bd40891a870987087ca285dbc6a7d8c42d19d2790b6328b30cac38f"
)
STATE_COMPATIBILITY_DIGEST = (
    "225b26aa40fa252a93288f525a4a36b071a4e56298480ba9c7d6dd50e1b3225c"
)
RESEARCH_MANIFEST_SHA256 = (
    "e75c7114818bec1f2a455143afe240fa31c9bf75282b6659b11eb49df8c347a2"
)
ORDER = (("OFF", "ON"), ("ON", "OFF"), ("OFF", "ON"))
REQUESTS = 8
PROMPT = [10, 20, 30, 40]
OUTPUT = [40, 40]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def request_json(url: str, payload: dict | None = None, timeout: int = 10) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_ready(base_url: str, process: subprocess.Popen[str]) -> float:
    started = time.perf_counter()
    while time.perf_counter() - started < 15:
        if process.poll() is not None:
            raise RuntimeError(f"service exited during startup: {process.returncode}")
        try:
            if request_json(base_url + "/health", timeout=1).get("status") == "ok":
                return time.perf_counter() - started
        except Exception:
            time.sleep(0.02)
    raise TimeoutError("StateAxis service did not become ready")


def stop_service(process: subprocess.Popen[str]) -> tuple[int, bool]:
    process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=10), False
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            return process.wait(timeout=5), True
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return process.wait(timeout=5), True


def configure_manager(root: Path, enabled: bool) -> Path:
    state = root / ("manager-on" if enabled else "manager-off") / "config.json"
    environment = os.environ.copy()
    environment["VLLM_HUST_EXT_CONFIG"] = str(state)
    subprocess.run(
        [
            str(CLI),
            "extension",
            "configure",
            "org.vllm-hust.stateaxis-automatic-prefix-cache",
            "--file",
            str(CONFIGURE),
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    if enabled:
        subprocess.run(
            [
                str(CLI),
                "extension",
                "enable",
                "org.vllm-hust.stateaxis-automatic-prefix-cache",
            ],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    check = subprocess.run(
        [
            str(CLI),
            "extension",
            "check",
            "org.vllm-hust.stateaxis-automatic-prefix-cache",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    status = json.loads(check.stdout)
    assert ("enabled" in status["states"]) is enabled
    return state


def engine_command(port: int) -> list[str]:
    digest = {
        "weights": "11" * 32,
        "rope": "22" * 32,
        "state": "33" * 32,
        "cache": "44" * 32,
        "scheduler": "55" * 32,
        "compiled": "66" * 32,
    }
    worker = (
        "NATIVE_MOCK_ACTIVE_STATES=0 NATIVE_MOCK_STATE_CAPACITY=8 "
        "NATIVE_MOCK_UNIQUE_FORK_IDS=1 python3 "
        f"{STATEAXIS / 'backend/native_protocol_mock_worker.py'}"
    )
    return [
        str(ENGINE),
        "--listen",
        f"127.0.0.1:{port}",
        "--native-command",
        worker,
        "--weights-digest",
        digest["weights"],
        "--rope-semantics-digest",
        digest["rope"],
        "--state-schema-digest",
        digest["state"],
        "--cache-layout-digest",
        digest["cache"],
        "--scheduler-plan-digest",
        digest["scheduler"],
        "--compiled-plan-digest",
        digest["compiled"],
        "--execution-config",
        str(ARTIFACT / "execution_config.bin"),
        "--execution-config-sha256",
        EXECUTION_CONFIG_SHA256,
        "--execution-artifact-identity-sha256",
        EXECUTION_ARTIFACT_SHA256,
        "--decode-mlp-policy",
        "fused_gate_up_aclnn_swiglu",
        "--prefix-extension-mode",
        "paged_append_prefill",
        "--append-prefill-max-batch-size",
        "1",
        "--max-batch-size",
        "1",
        "--batch-window-us",
        "0",
    ]


def run_arm(pair: int, position: int, arm: str, manager_config: Path) -> dict:
    run_id = f"pair-{pair + 1}-{position + 1}-{arm.lower()}"
    log_path = ROOT / f"{run_id}.log"
    result_path = ROOT / f"{run_id}.json"
    if log_path.exists() or result_path.exists():
        raise FileExistsError(f"refusing to overwrite {run_id} evidence")
    port = free_port()
    environment = os.environ.copy()
    environment.update(
        {
            "VLLM_HUST_EXT_CONFIG": str(manager_config),
            "RUST_LOG": "info",
        }
    )
    command = [
        str(CLI),
        "run",
        "--shutdown-grace-seconds",
        "10",
        "--",
        *engine_command(port),
    ]
    responses = []
    started = time.perf_counter()
    forced_stop = False
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=STATEAXIS,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            base_url = f"http://127.0.0.1:{port}"
            startup_seconds = wait_ready(base_url, process)
            metrics_before = request_json(base_url + "/metrics")
            executor_before = request_json(base_url + "/v1/native/executor_stats")
            workload_started = time.perf_counter()
            for index in range(REQUESTS):
                response = request_json(
                    base_url + "/v1/native/generate",
                    {
                        "request_id": 10_000 + pair * 100 + position * 20 + index,
                        "namespace": {
                            "scope": "apc-ecpa-stable-prefix",
                            "model_revision": "mock-qwen2.5-14b-contract",
                            "quality_epoch": "greedy",
                            "state_compatibility_digest": STATE_COMPATIBILITY_DIGEST,
                        },
                        "input": {"kind": "cold", "token_ids": PROMPT},
                        "max_new_tokens": len(OUTPUT),
                        "retain_state": False,
                        "cache_salt": "stable-repeated-prefix",
                        "deadline_ms": None,
                    },
                )
                responses.append(response)
            workload_seconds = time.perf_counter() - workload_started
            metrics_after = request_json(base_url + "/metrics")
            executor_after = request_json(base_url + "/v1/native/executor_stats")
        finally:
            if process.poll() is None:
                return_code, forced_stop = stop_service(process)
            else:
                return_code = int(process.returncode)
    elapsed_seconds = time.perf_counter() - started
    result = {
        "schema_version": "stateaxis-apc-ecpa-matched-process/v1",
        "evidence_label": "matched_process_mock_non_performance",
        "run_id": run_id,
        "pair": pair + 1,
        "position": position + 1,
        "arm": arm,
        "manager_config": str(manager_config),
        "command": command,
        "startup_seconds": startup_seconds,
        "workload_seconds": workload_seconds,
        "elapsed_seconds": elapsed_seconds,
        "service_return_code": return_code,
        "forced_stop": forced_stop,
        "responses": responses,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "executor_before": executor_before,
        "executor_after": executor_after,
        "performance_claim": False,
        "accelerator_execution": False,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def metric_delta(result: dict, name: str) -> int:
    return int(result["metrics_after"].get(name, 0)) - int(
        result["metrics_before"].get(name, 0)
    )


def validate(result: dict) -> None:
    enabled = result["arm"] == "ON"
    responses = result["responses"]
    assert len(responses) == REQUESTS
    assert all(row["output_token_ids"] == OUTPUT for row in responses)
    assert result["forced_stop"] is False
    assert result["metrics_before"]["automatic_prefix_cache_enabled"] is enabled
    assert result["metrics_after"]["automatic_prefix_cache_enabled"] is enabled
    assert result["executor_before"]["active_states"] == 0
    expected_active = 1 if enabled else 0
    assert result["executor_after"]["active_states"] == expected_active
    counters = {
        "automatic_prefix_cache_lookups": REQUESTS if enabled else 0,
        "automatic_prefix_cache_hits": REQUESTS - 1 if enabled else 0,
        "automatic_prefix_cache_reused_tokens": (
            len(PROMPT) * (REQUESTS - 1) if enabled else 0
        ),
        "automatic_prefix_cache_inserts": 1 if enabled else 0,
        "automatic_prefix_cache_evictions": 0,
        "automatic_prefix_cache_lease_blocked_inserts": 0,
        "automatic_prefix_cache_hash_collisions": 0,
        "automatic_prefix_cache_stale_entries": 0,
    }
    for name, expected in counters.items():
        assert metric_delta(result, name) == expected, (name, result["run_id"])
    if enabled:
        assert responses[0]["hit_kind"] == "cold"
        assert responses[0]["reused_tokens"] == 0
        assert all(row["hit_kind"] == "exact" for row in responses[1:])
        assert all(row["reused_tokens"] == len(PROMPT) for row in responses[1:])
        identity = result["metrics_after"]["automatic_prefix_cache_mod"]
        assert identity == {
            "mod_id": "org.vllm-hust.stateaxis-automatic-prefix-cache",
            "version": "0.2.0",
            "manifest_sha256": RESEARCH_MANIFEST_SHA256,
            "performance_qualified": False,
            "experiment_mode": True,
        }
        assert result["metrics_after"]["automatic_prefix_cache_entries"] == 1
    else:
        assert all(row["hit_kind"] == "cold" for row in responses)
        assert all(row["reused_tokens"] == 0 for row in responses)
        assert result["metrics_after"]["automatic_prefix_cache_mod"] is None
        assert result["metrics_after"]["automatic_prefix_cache_entries"] == 0


def main() -> None:
    required = [
        CLI,
        ENGINE,
        CONFIGURE,
        ARTIFACT / "execution_config.bin",
        MOD_REPO / "RESEARCH_MANIFEST.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing prerequisites: {missing}")
    expected_hashes = {
        ARTIFACT / "execution_config.bin": EXECUTION_CONFIG_SHA256,
        MOD_REPO / "RESEARCH_MANIFEST.json": RESEARCH_MANIFEST_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"digest drift: {path}: {actual} != {expected}")
    existing = list(ROOT.glob("pair-*.json")) + list(ROOT.glob("pair-*.log"))
    if existing or (ROOT / "SUMMARY.json").exists():
        raise SystemExit("refusing to overwrite existing campaign evidence")

    with tempfile.TemporaryDirectory(prefix="apc-ecpa-manager-") as temporary:
        manager_root = Path(temporary)
        on_config = configure_manager(manager_root, True)
        off_config = configure_manager(manager_root, False)
        runs = []
        for pair, arms in enumerate(ORDER):
            for position, arm in enumerate(arms):
                runs.append(
                    run_arm(
                        pair,
                        position,
                        arm,
                        on_config if arm == "ON" else off_config,
                    )
                )
    for result in runs:
        validate(result)
    summary = {
        "schema_version": "stateaxis-apc-ecpa-matched-process-summary/v1",
        "evidence_label": "matched_process_mock_non_performance",
        "order": ORDER,
        "pairs": len(ORDER),
        "runs": len(runs),
        "requests_per_run": REQUESTS,
        "all_outputs_exact": True,
        "all_lifecycle_checks_passed": True,
        "off": {
            "runs": 3,
            "lookups_per_run": 0,
            "hits_per_run": 0,
            "reused_tokens_per_run": 0,
            "final_entries_per_run": 0,
        },
        "on": {
            "runs": 3,
            "lookups_per_run": REQUESTS,
            "hits_per_run": REQUESTS - 1,
            "reused_tokens_per_run": len(PROMPT) * (REQUESTS - 1),
            "inserts_per_run": 1,
            "final_entries_per_run": 1,
        },
        "stateaxis_commit": git(STATEAXIS, "rev-parse", "HEAD"),
        "stateaxis_dirty": bool(git(STATEAXIS, "status", "--porcelain")),
        "mod_commit": git(MOD_REPO, "rev-parse", "HEAD"),
        "mod_dirty": bool(git(MOD_REPO, "status", "--porcelain")),
        "manager_commit": git(MANAGER_REPO, "rev-parse", "HEAD"),
        "manager_dirty": bool(git(MANAGER_REPO, "status", "--porcelain")),
        "research_manifest_sha256": RESEARCH_MANIFEST_SHA256,
        "execution_config_sha256": EXECUTION_CONFIG_SHA256,
        "execution_artifact_identity_sha256": EXECUTION_ARTIFACT_SHA256,
        "accelerator_execution": False,
        "performance_claim": False,
        "result": "pass",
    }
    (ROOT / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    evidence_files = sorted(
        path
        for path in ROOT.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS"}
    )
    (ROOT / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in evidence_files)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
