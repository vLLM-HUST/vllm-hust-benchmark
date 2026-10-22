#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""W2b ``fi_sampling`` e2e A/B refresh on the new baseline (2026-09-10).

Purpose
-------
The original W2b evidence (``docs/evidence/w2b-fi-sampling/REPORT.md``) is bound
to the OLD baseline (vllm 0.23.0 / CANN 9.0.1 / triton-ascend 3.2.1) and the old
host source trees are gone; the REPORT itself says its numbers "不跨宿主继承".
This harness re-runs the same protocol on the CURRENT baseline and emits the
numbers a catalog ``qualified`` decision needs.

Protocol source of truth: the W2b REPORT (``§3.1`` 口径 + ``run_e2e_matrix.sh``),
with the two new-baseline bits added **symmetrically to both arms**:

  * ``--compilation-config {"cudagraph_mode":"FULL"}`` (宿主缺陷规避, pitfalls
    §1.3: the default FULL_AND_PIECEWISE crashes engine init on this baseline);
  * ``VLLM_DISABLE_COMPILE_CACHE=1``.

What is measured
----------------
Per leg (one independent server lifecycle): ``vllm bench serve`` openai-chat,
random dataset, ``--ignore-eos``, fixed lengths, TPOT/TTFT/ITL percentiles.  The
instrumented quantity is the client-reported ``median_tpot_ms`` (the per-step
time proxy the REPORT uses).  Arms: OFF = no env; ON = ``VLLM_HUST_FI_SAMPLING=1``
(+ ``VLLM_HUST_FI_SAMPLING_TRACE=1`` for the route histogram).  Leg order is the
pre-registered alternating one: off_a, on_a, off_b, on_b, off_c, on_c.

Two cases, copied verbatim from the W2b protocol:

  * ``api1``  (untruncated sampling -- the main win region): max-model-len 4096,
    max-num-seqs 512, random 1024 in / 256 out, 256 prompts, concurrency 64,
    NO top-k/top-p.  ``--generation-config vllm`` is required: the model's own
    ``generation_config.json`` would otherwise override the served defaults with
    top_k=20/top_p=0.8 and silently turn this into a truncated (fork) leg.
  * ``joint`` (k=50/p=0.95, large concurrency): max-model-len 512, 128 in / 64
    out, 512 prompts, concurrency 512.  One ON leg only, purely to re-test the
    ``B >= 256`` reachability fact (§3.4 of the REPORT) -- no TPOT attribution.

Outputs (all under ``bench/results/``)
--------------------------------------
``fisampling_refresh_e2e.json``        pre-registration meta + per-leg rows
                                       (written incrementally) + summary;
``logs_fisampling_refresh/``           one server log + one client log per leg,
                                       plus the raw ``vllm bench serve`` JSON.

Usage
-----
    bash bench/bench_fisampling_refresh_e2e.sh              # driver (card 7)
    python3 bench/bench_fisampling_refresh_e2e.py           # orchestrator
    python3 bench/bench_fisampling_refresh_e2e.py --leg on on_a 8352 api1   # one child
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# --------------------------------------------------------------------------
# Pre-registered constants (written into the JSON ``meta`` before any leg runs).
# --------------------------------------------------------------------------
MODEL = "/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct"
SERVED_NAME = "qwen14b"
COMPILATION_CONFIG = '{"cudagraph_mode":"FULL"}'
GPU_MEM_UTIL = "0.85"

#: api1 (untruncated) -- W2b REPORT §3.1/§3.2 + run_e2e_matrix.sh.
API1_SERVER_ARGS = ["--max-model-len", "4096", "--max-num-seqs", "512"]
API1_CLIENT_ARGS = [
    "--dataset-name",
    "random",
    "--random-input-len",
    "1024",
    "--random-output-len",
    "256",
    "--num-prompts",
    "256",
    "--max-concurrency",
    "64",
]

#: joint -- W2b REPORT §3.3/§3.4 + run_e2e_matrix.sh (reachability only).
JOINT_SERVER_ARGS = ["--max-model-len", "512", "--max-num-seqs", "512"]
JOINT_CLIENT_ARGS = [
    "--dataset-name",
    "random",
    "--random-input-len",
    "128",
    "--random-output-len",
    "64",
    "--num-prompts",
    "512",
    "--max-concurrency",
    "512",
    "--top-k",
    "50",
    "--top-p",
    "0.95",
]

#: api1 arms: alternating order (absorbs slow machine drift).
LEGS = [("off", "a"), ("on", "a"), ("off", "b"), ("on", "b"), ("off", "c"), ("on", "c")]
#: joint: reachability probe only (single ON leg, trace on).
JOINT_LEGS = [("on", "j")]

PORT_BASE = 8351
LEG_SLEEP_S = 10
SERVER_READY_TIMEOUT_S = 1800
CLIENT_TIMEOUT_S = 3600

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"
LOG_DIR = RESULTS / "logs_fisampling_refresh"
OUT_JSON = RESULTS / "fisampling_refresh_e2e.json"

SENTINEL = "@@FISAMP_LEG_JSON@@"
TRACE_LIMIT_ROWS = 5

#: pre-registered decision rule + projection, frozen before the run.
PREREGISTRATION = {
    "date": "2026-09-10",
    "subject": "W2b fi_sampling e2e evidence refresh on the new baseline",
    "arms": {"OFF": "no env (default-off)", "ON": "VLLM_HUST_FI_SAMPLING=1"},
    "cases": {
        "api1": "untruncated sampling (main win region), 3 ON / 3 OFF alternating",
        "joint": "k=50/p=0.95 large-concurrency reachability probe, 1 ON leg",
    },
    "decision_rule": (
        "delta_frac = (median_on - median_off) / median_off over the 3 repeats per "
        "arm (negative => ON faster, the W2b sign convention); spread_arm = "
        "(max - min) / median within an arm; noise_band = max(spread_off, "
        "spread_on). |delta_frac| < noise_band => NOISE-BAND / NOT DECIDABLE "
        "(same口径 as W2b). Otherwise WIN (delta_frac < 0) / LOSS (delta_frac > 0)."
    ),
    "projection": (
        "W2 single-op bench @B=64: fork chain ~1.27 ms vs FI api1 ~0.35 ms/step "
        "=> ~0.9 ms/step saved; the new-baseline TPOT median measured here sets "
        "the projector. If sampling is still ~1% of step time the expected |Δ| "
        "stays inside the run-to-run noise band (W2b precedent)."
    ),
    "health_gates": [
        "0 TypeError lines in every leg log",
        "OFF legs: zero 'fi_sampling sampling path is ACTIVE' lines (and zero "
        "fi_sampling runtime traces)",
        "ON legs: >=1 ACTIVE line and a fi_api1 route histogram for api1",
        "port slot free before and after every leg",
    ],
    "no_retuning": "no re-tuning / re-running to chase significance; extra "
    "repeats require a new pre-registration",
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _read_kv(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}=(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def version_triple() -> dict:
    """vllm / vllm-ascend / torch / CANN versions, read from dists + files."""
    import importlib.metadata as md

    def dist(name: str) -> str:
        try:
            return md.version(name)
        except Exception:  # noqa: BLE001 -- a missing dist must not kill a leg
            return "unavailable"

    cann: dict = {"version": "unavailable"}
    roots = [
        os.environ.get("ASCEND_HOME_PATH"),
        "/usr/local/ascend91/Ascend/cann-9.1.0",
    ]
    for root in roots:
        if not root:
            continue
        info = Path(root) / "aarch64-linux" / "ascend_toolkit_install.info"
        if not info.is_file():
            continue
        text = info.read_text(errors="replace")
        cann = {
            "version": _read_kv(text, "version") or "unavailable",
            "innerversion": _read_kv(text, "innerversion") or "unavailable",
            "path": str(root),
        }
        break
    return {
        "vllm": dist("vllm"),
        "vllm_ascend": dist("vllm-ascend"),
        "torch": dist("torch"),
        "torch_npu": dist("torch-npu"),
        "triton_ascend": dist("triton-ascend"),
        "cann": cann,
    }


def git_probe(repo: Path) -> dict:
    def git(*args: str) -> str:
        try:
            proc = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:  # noqa: BLE001
            return ""
        return proc.stdout.strip()

    head = git("rev-parse", "HEAD")
    porcelain = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]
    untracked = [ln for ln in porcelain if ln.startswith("??")]
    modified_tracked = [ln for ln in porcelain if not ln.startswith("??")]
    return {
        "head": head,
        "short": head[:7],
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked": bool(modified_tracked),
        "modified_tracked": modified_tracked,
        "untracked": untracked,
        "subject": git("log", "-1", "--pretty=%s"),
    }


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _vllm_exe() -> str:
    found = shutil.which("vllm")
    if found:
        return found
    return sys.executable


def parse_histograms(text: str) -> dict:
    """Route traces / histogram / health counters grepped from a leg log."""
    route_counts: dict[str, int] = {}
    for decision, _batch in re.findall(r"route=([a-z_0-9]+) B=([0-9]+)", text):
        route_counts[decision] = route_counts.get(decision, 0) + 1

    final_hist: dict[str, int] | None = None
    max_b: int | None = None
    for m in re.finditer(
        r"\[fi-sampling histogram calls=([0-9]+) max_B=([0-9]+)\] (\{[^}]*\})", text
    ):
        max_b = int(m.group(2))
        final_hist = json.loads(m.group(3).replace("'", '"'))
    if final_hist is None:
        tail = re.findall(r"\[fi-sampling histogram\] (\{[^}]*\})", text)
        if tail:
            final_hist = json.loads(tail[-1].replace("'", '"'))

    return {
        "active_lines": text.count("fi_sampling sampling path is ACTIVE"),
        "fi_sampling_mentions": text.count("fi_sampling"),
        "type_error_lines": text.count("TypeError"),
        "route_trace_counts": route_counts,
        "route_histogram": final_hist,
        "max_B": max_b,
    }


def _wait_health(port: int, proc: subprocess.Popen, log_path: Path) -> float:
    """Block until /health answers; return seconds waited.  Raises on death."""
    t0 = time.monotonic()
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() - t0 < SERVER_READY_TIMEOUT_S:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-3000:]
            raise RuntimeError(
                f"server exited rc={proc.returncode} during startup\n{tail}"
            )
        try:
            with urllib.request.urlopen(url, timeout=2):
                return time.monotonic() - t0
        except (urllib.error.URLError, OSError):
            time.sleep(5)
    raise RuntimeError("server health timeout")


def _stop_server(proc: subprocess.Popen, port: int) -> bool:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=180)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=60)
    for _ in range(120):
        if port_is_free(port):
            return True
        time.sleep(1)
    return port_is_free(port)


# --------------------------------------------------------------------------
# child: one leg = server + client, one process
# --------------------------------------------------------------------------
def run_leg(kind: str, mode: str, leg_tag: str, port: int) -> int:
    """One leg = one server lifecycle + one client run, in this process.

    ``leg_tag`` is the fully qualified ``<mode>_<tag>`` (e.g. ``off_a``,
    ``on_b``): it names every artifact so that no two legs can ever share a
    file (an earlier revision used the bare tag and silently overwrote the OFF
    logs with the ON logs of the same index).
    """
    if kind == "api1":
        server_args, client_args = API1_SERVER_ARGS, API1_CLIENT_ARGS
    else:
        server_args, client_args = JOINT_SERVER_ARGS, JOINT_CLIENT_ARGS

    if mode == "on":
        os.environ["VLLM_HUST_FI_SAMPLING"] = "1"
        os.environ["VLLM_HUST_FI_SAMPLING_TRACE"] = "1"
    else:
        os.environ.pop("VLLM_HUST_FI_SAMPLING", None)
        os.environ.pop("VLLM_HUST_FI_SAMPLING_TRACE", None)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    server_log = LOG_DIR / f"serve_{kind}_{leg_tag}.log"
    client_log = LOG_DIR / f"client_{kind}_{leg_tag}.log"
    result_name = f"serve_{kind}_{leg_tag}.json"

    vllm = _vllm_exe()
    server_cmd = [
        vllm,
        "serve",
        MODEL,
        "--served-model-name",
        SERVED_NAME,
        "--generation-config",
        "vllm",
        "--gpu-memory-utilization",
        GPU_MEM_UTIL,
        "--compilation-config",
        COMPILATION_CONFIG,
        "--port",
        str(port),
        *server_args,
    ]
    client_cmd = [
        vllm,
        "bench",
        "serve",
        "--backend",
        "openai-chat",
        "--model",
        SERVED_NAME,
        "--tokenizer",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--endpoint",
        "/v1/chat/completions",
        "--ignore-eos",
        "--num-warmups",
        "8",
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--metric-percentiles",
        "50,95,99",
        "--save-result",
        "--result-dir",
        str(LOG_DIR),
        "--result-filename",
        result_name,
        *client_args,
    ]

    print(
        f"[leg {kind}_{leg_tag}] begin mode={mode} port={port} pid={os.getpid()}\n"
        f"[leg {kind}_{leg_tag}] server: {' '.join(server_cmd)}",
        flush=True,
    )
    t_proc = time.monotonic()
    with open(server_log, "w") as fh:
        server = subprocess.Popen(
            server_cmd, stdout=fh, stderr=subprocess.STDOUT, cwd="/tmp"
        )
    row: dict = {
        "kind": kind,
        "leg": leg_tag,
        "mode": mode,
        "port": port,
        "ok": False,
        "server_log": str(server_log.relative_to(REPO)),
        "client_log": str(client_log.relative_to(REPO)),
        "result_json": str((LOG_DIR / result_name).relative_to(REPO)),
    }
    try:
        row["ready_s"] = round(_wait_health(port, server, server_log), 1)
        print(f"[leg {kind}_{leg_tag}] server ready in {row['ready_s']}s", flush=True)

        server_text = server_log.read_text(errors="replace")
        active = server_text.count("fi_sampling sampling path is ACTIVE")
        print(f"[leg {kind}_{leg_tag}] ACTIVE lines = {active}", flush=True)
        if mode == "on" and active < 1:
            raise RuntimeError("FI plugin did not activate in the ON leg")
        if mode == "off" and active != 0:
            raise RuntimeError("FI plugin activated in the OFF leg")

        t_client = time.monotonic()
        with open(client_log, "w") as fh:
            client = subprocess.run(
                client_cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                cwd="/tmp",
                timeout=CLIENT_TIMEOUT_S,
            )
        row["client_wall_s"] = round(time.monotonic() - t_client, 1)
        row["client_rc"] = client.returncode

        result_path = LOG_DIR / result_name
        if result_path.is_file():
            data = json.loads(result_path.read_text())
            row["metrics"] = {
                key: data.get(key)
                for key in (
                    "median_tpot_ms",
                    "mean_tpot_ms",
                    "p95_tpot_ms",
                    "p99_tpot_ms",
                    "median_ttft_ms",
                    "output_throughput",
                    "completed",
                    "failed",
                    "num_prompts",
                    "max_concurrency",
                    "duration",
                )
            }
            row["ok"] = client.returncode == 0 and bool(data.get("completed"))
        else:
            row["metrics"] = None
    finally:
        row["port_free_after"] = _stop_server(server, port)

    row["total_wall_s"] = round(time.monotonic() - t_proc, 1)
    print(SENTINEL + json.dumps(row), flush=True)
    return 0 if row.get("ok") else 1


# --------------------------------------------------------------------------
# parent: orchestrator
# --------------------------------------------------------------------------
def write_results(payload: dict) -> None:
    payload["meta"]["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def summarize(rows: list[dict]) -> dict:
    tpot: dict[str, list[float]] = {"off": [], "on": []}
    for row in rows:
        if row.get("ok") and row.get("metrics"):
            tpot[row["mode"]].append(row["metrics"]["median_tpot_ms"])

    out: dict = {
        "n_legs": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "health": {
            "type_error_lines_total": sum(
                r.get("traces", {}).get("type_error_lines", 0) for r in rows
            ),
            "off_legs_active_lines": [
                r.get("traces", {}).get("active_lines", 0)
                for r in rows
                if r["mode"] == "off"
            ],
            "on_legs_active_lines": [
                r.get("traces", {}).get("active_lines", 0)
                for r in rows
                if r["mode"] == "on"
            ],
            "all_ports_free_after": all(r.get("port_free_after") for r in rows),
        },
    }
    for mode in ("off", "on"):
        vals = sorted(tpot[mode])
        if not vals:
            continue
        med = statistics.median(vals)
        out[f"{mode}_tpot_medians_ms"] = [round(v, 2) for v in vals]
        out[f"{mode}_tpot_median_ms"] = round(med, 3)
        out[f"{mode}_tpot_spread_ms"] = round(max(vals) - min(vals), 3)
        out[f"{mode}_tpot_spread_frac"] = round((max(vals) - min(vals)) / med, 5)

    if "off_tpot_median_ms" in out and "on_tpot_median_ms" in out:
        off_med = out["off_tpot_median_ms"]
        on_med = out["on_tpot_median_ms"]
        delta_frac = (on_med - off_med) / off_med
        band = max(out["off_tpot_spread_frac"], out["on_tpot_spread_frac"])
        out["delta_median_ms"] = round(on_med - off_med, 3)
        out["delta_median_pct"] = round(100.0 * delta_frac, 3)
        out["noise_band_frac"] = round(band, 5)
        out["noise_band_pct"] = round(100.0 * band, 3)
        out["within_noise_band"] = bool(abs(delta_frac) < band)
        if out["within_noise_band"]:
            out["verdict"] = (
                "NOISE-BAND / NOT DECIDABLE (|Δmedian| < max(spread_off, spread_on))"
            )
        else:
            out["verdict"] = (
                "WIN (ON faster) " if delta_frac < 0 else "LOSS (ON slower) "
            ) + f"Δmedian={100.0 * delta_frac:+.3f}%"
        # Projection: W2 single-op bench @B=64 saves ~0.9 ms/step.
        out["projection"] = {
            "saved_ms_per_step_w2_bench_b64": 0.9,
            "projected_delta_pct": round(-100.0 * 0.9 / off_med, 3),
            "note": "|-0.9/TPOT| is the analytical ceiling of the e2e effect",
        }
    return out


def print_summary(payload: dict) -> None:
    print("\n" + "=" * 78, flush=True)
    print("W2b fi_sampling e2e refresh -- summary", flush=True)
    print("=" * 78, flush=True)
    print(
        f"{'leg':<10}{'mode':<6}{'median_tpot':>12}{'p95':>9}{'p99':>9}"
        f"{'out_tput':>10}{'ok':>4}{'ACTIVE':>7}{'route_hist':>28}",
        flush=True,
    )
    for row in payload["legs"]:
        m = row.get("metrics") or {}
        tr = row.get("traces", {})
        print(
            f"{row['kind'] + '_' + row['leg']:<10}{row['mode']:<6}"
            f"{m.get('median_tpot_ms', float('nan')):>12.2f}"
            f"{m.get('p95_tpot_ms', float('nan')):>9.2f}"
            f"{m.get('p99_tpot_ms', float('nan')):>9.2f}"
            f"{m.get('output_throughput', float('nan')):>10.2f}"
            f"{str(row.get('ok')):>4}"
            f"{tr.get('active_lines', 0):>7}"
            f"{str(tr.get('route_histogram')):>28}",
            flush=True,
        )
    print("-" * 78, flush=True)
    for s in payload.get("summaries", []):
        print(f"### case {s['case']}", flush=True)
        for key, val in s.items():
            if key in ("case",):
                continue
            print(f"  {key} = {val}", flush=True)
    print("=" * 78 + "\n", flush=True)


def orchestrate() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    self_path = str(Path(__file__).resolve())
    print(
        f"[orch] fi_sampling refresh A/B -> {OUT_JSON}\n"
        f"[orch] card={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')} "
        f"cwd={os.getcwd()} logs={LOG_DIR}",
        flush=True,
    )

    payload: dict = {
        "experiment": "W2b fi_sampling e2e A/B refresh (new baseline 2026-09-10)",
        "meta": {
            "preregistration": PREREGISTRATION,
            "baseline": version_triple(),
            "plugin_git": git_probe(REPO),
            "env_at_launch": {
                "ASCEND_RT_VISIBLE_DEVICES": os.environ.get(
                    "ASCEND_RT_VISIBLE_DEVICES", "<unset>"
                ),
                "VLLM_DISABLE_COMPILE_CACHE": "1 (forced per leg)",
                "HF_HUB_OFFLINE": "1 (forced per leg)",
            },
        },
        "fixture": {
            "server": {
                "model": MODEL,
                "served_model_name": SERVED_NAME,
                "generation_config": "vllm",
                "gpu_memory_utilization": GPU_MEM_UTIL,
                "compilation_config": COMPILATION_CONFIG,
            },
            "api1": {
                "server_args": API1_SERVER_ARGS,
                "client_args": API1_CLIENT_ARGS,
                "leg_order": [f"{m}_{t}" for m, t in LEGS],
                "port_slots": [PORT_BASE + i for i in range(len(LEGS))],
            },
            "joint": {
                "server_args": JOINT_SERVER_ARGS,
                "client_args": JOINT_CLIENT_ARGS,
                "leg_order": [f"{m}_{t}" for m, t in JOINT_LEGS],
            },
            "note": "offline? NO -- this is the W2b serve protocol "
            "(vllm bench serve, openai-chat), kept verbatim except for the "
            "graph-mode + cache-disable pair, applied to both arms.",
        },
        "legs": [],
        "summaries": [],
    }
    write_results(payload)  # pre-registration is on disk before leg 1

    cases = [("api1", LEGS), ("joint", JOINT_LEGS)]
    for case, legs in cases:
        case_rows: list[dict] = []
        for idx, (mode, tag) in enumerate(legs):
            leg_tag = f"{mode}_{tag}"
            port = PORT_BASE + idx + (0 if case == "api1" else 100)
            env = dict(os.environ)
            env["HF_HUB_OFFLINE"] = "1"
            env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
            env.pop("VLLM_HUST_FI_SAMPLING", None)
            env.pop("VLLM_HUST_FI_SAMPLING_TRACE", None)

            free_before = port_is_free(port)
            print(
                f"[orch] case={case} leg {idx + 1}/{len(legs)} {leg_tag} "
                f"port={port} free_before={free_before} "
                f"start={time.strftime('%H:%M:%S')}",
                flush=True,
            )
            leg_log = LOG_DIR / f"leg_{case}_{leg_tag}.log"
            rc: object = -1
            with open(leg_log, "w") as fh:
                proc = subprocess.Popen(
                    [
                        sys.executable,
                        self_path,
                        "--leg",
                        mode,
                        leg_tag,
                        str(port),
                        case,
                    ],
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    cwd="/tmp",
                    env=env,
                )
                leg_timeout = SERVER_READY_TIMEOUT_S + CLIENT_TIMEOUT_S + 600
                try:
                    rc = proc.wait(timeout=leg_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    rc = "timeout"

            text = leg_log.read_text(errors="replace")
            row = None
            for line in text.splitlines():
                if line.startswith(SENTINEL):
                    try:
                        row = json.loads(line[len(SENTINEL) :])
                    except json.JSONDecodeError:
                        continue
            if row is None:
                row = {
                    "kind": case,
                    "leg": leg_tag,
                    "mode": mode,
                    "port": port,
                    "ok": False,
                }
            row["index"] = idx + 1
            row["subprocess_rc"] = rc
            row["ok"] = bool(row.get("ok")) and rc == 0
            row["port_free_before"] = free_before
            row["traces"] = parse_histograms(text)
            row["leg_log"] = str(leg_log.relative_to(REPO))
            server_log = LOG_DIR / f"serve_{case}_{leg_tag}.log"
            if server_log.is_file():
                row["traces"] = parse_histograms(server_log.read_text(errors="replace"))
                row["traces"]["leg_log_fi_sampling_mentions"] = text.count(
                    "fi_sampling"
                )
                row["traces"]["leg_log_type_error_lines"] = text.count("TypeError")
            payload["legs"].append(row)
            case_rows.append(row)

            m = row.get("metrics") or {}
            print(
                f"[orch] case={case} {leg_tag} done rc={rc} ok={row['ok']} "
                f"median_tpot={m.get('median_tpot_ms')} "
                f"ACTIVE={row['traces']['active_lines']} "
                f"hist={row['traces']['route_histogram']}",
                flush=True,
            )
            write_results(payload)  # incremental: survive an aborted run

            if idx < len(legs) - 1:
                print(f"[orch] sleep {LEG_SLEEP_S}s", flush=True)
                time.sleep(LEG_SLEEP_S)

        summary = summarize(case_rows)
        summary["case"] = case
        payload["summaries"] = [
            s for s in payload["summaries"] if s.get("case") != case
        ] + [summary]
        write_results(payload)

    print_summary(payload)
    print(f"[orch] DONE -> {OUT_JSON}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="W2b fi_sampling e2e refresh A/B harness"
    )
    parser.add_argument("--leg", action="store_true", help="run a single child leg")
    parser.add_argument("mode", nargs="?", choices=["off", "on"])
    parser.add_argument("tag", nargs="?", help="<mode>_<tag>, e.g. off_a")
    parser.add_argument("port", nargs="?", type=int)
    parser.add_argument("case", nargs="?", choices=["api1", "joint"])
    args = parser.parse_args()
    if args.leg:
        if not (args.mode and args.tag and args.port and args.case):
            parser.error("--leg requires: <mode> <tag> <port> <case>")
        return run_leg(args.case, args.mode, args.tag, args.port)
    return orchestrate()


if __name__ == "__main__":
    sys.exit(main())
