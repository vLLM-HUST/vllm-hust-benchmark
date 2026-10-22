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

"""W3.1 gelu e2e A/B (6 legs, NPU card 7) -- pre-registered in
``bench/results/EXPECTATIONS-w31-e2e.md`` (read that file first; every knob
below is copied from it and must not be re-tuned after the fact).

Fixture (identical for both arms)
--------------------------------
Offline ``LLM.generate``, one independent process per leg.  64 prompts of
~256 tokens each, ``SamplingParams(temperature=0, max_tokens=128,
ignore_eos=True)`` -> 64x128 = 8192 output tokens per leg (asserted).  The
instrumented quantity is ``time.perf_counter`` wrapped around ``llm.generate``.
Engine kwargs mirror ``knowledge/evidence/cascade/ev2_run.py`` for everything
the pre-registration leaves unspecified; the pre-registered knobs
(``max_model_len=4096``, ``max_num_seqs=64``, ``gpu_memory_utilization=0.85``,
``compilation_config={"cudagraph_mode": "FULL"}``) are applied verbatim.
``--compilation-config`` is passed through the offline API's
``compilation_config`` keyword (the pre-registration writes the serve CLI
spelling).

Arms
----
OFF = ``VLLM_HUST_FI_GELU`` unset; ON = ``VLLM_HUST_FI_GELU=1``.  The env is
set in the leg *child* process before ``vllm`` is imported.  Leg order is the
pre-registered alternating one: off_a, on_a, off_b, on_b, off_c, on_c.

Per-leg bookkeeping: wall, ready time (process start -> ``LLM`` usable, split
into import / engine-init), env fingerprint, version triple (vllm /
vllm-ascend / CANN), plugin git commit + dirty flag, log-trace grep counts
(ON: ``fi_gelu activation path is ACTIVE``; OFF: zero ``fi_gelu`` traces).
Each leg also owns a distinct port slot (8341, 8342, ...) which is checked
free before/after the leg.  NOTE: the offline ``LLM`` API binds no HTTP
listener on this stack (single-process V1 uses a ZMQ IPC path), so the port
slot is a host-level guard / bookkeeping slot, not a vllm bind; the
pre-registered fixture is offline, so there is no serve port to release.

Outputs (all under ``bench/results/``)
--------------------------------------
``bench_w31_e2e.json``         per-leg rows (line-oriented, appended as legs
                               complete) + the mechanical summary;
``bench_w31_e2e_summary.txt``  the terminal summary table;
``logs_w31_e2e/leg_<tag>.log`` full stdout+stderr of each leg process.

Usage
-----
    python3 bench/bench_w31_e2e.py                     # orchestrator (6 legs)
    python3 bench/bench_w31_e2e.py --leg on on_a 8342  # single leg (child)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------
# Pre-registered constants (EXPECTATIONS-w31-e2e.md) -- do not tune.
# --------------------------------------------------------------------------
MODEL = "/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct"
N_PROMPTS = 64
MAX_TOKENS = 128
EXPECTED_OUT_TOKENS = N_PROMPTS * MAX_TOKENS  # 8192
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 64
GPU_MEM_UTIL = 0.85
COMPILATION_CONFIG = {"cudagraph_mode": "FULL"}
#: "The quick brown fox..." x 23 -> prompt ~254 tok (tokenizer-measured below).
SENT_REPEATS = 23
#: alternating leg order (absorbs slow machine drift; cascade rerun 7.3).
LEGS = [
    ("off", "a"),
    ("on", "a"),
    ("off", "b"),
    ("on", "b"),
    ("off", "c"),
    ("on", "c"),
]
PORT_BASE = 8341
LEG_SLEEP_S = 10
LEG_TIMEOUT_S = 3600

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "bench" / "results"
LOG_DIR = RESULTS / "logs_w31_e2e"
OUT_JSON = RESULTS / "bench_w31_e2e.json"
OUT_TXT = RESULTS / "bench_w31_e2e_summary.txt"

BASE_TEXT = "You are a helpful assistant. "
SENT = "The quick brown fox jumps over the lazy dog. "

#: sentinel that carries the per-leg JSON row out of the child process.
LEG_JSON_SENTINEL = "@@W31_LEG_JSON@@"

#: process start, used for the "process start -> LLM usable" ready time.
T_PROC_START = time.monotonic()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def build_prompts() -> list[str]:
    """64 equal-length prompts (~254 tok): shared prefix + tiny unique suffix."""
    prefix = BASE_TEXT + SENT * SENT_REPEATS
    return [
        prefix + f"\n\nQuestion {i}: What is {i}+{i}? Answer with the number only."
        for i in range(N_PROMPTS)
    ]


def port_is_free(port: int) -> bool:
    """True when nothing is listening/bound on ``127.0.0.1:port``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


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
        # "clean" means no tracked file is modified: this run only *adds*
        # untracked bench artifacts (bench/bench_w31_e2e.*, bench/results/*).
        "dirty": bool(modified_tracked),
        "modified_tracked": modified_tracked,
        "untracked": untracked,
        "subject": git("log", "-1", "--pretty=%s"),
    }


def env_fingerprint() -> dict:
    return {
        "VLLM_HUST_FI_GELU": os.environ.get("VLLM_HUST_FI_GELU", "<unset>"),
        "VLLM_DISABLE_COMPILE_CACHE": os.environ.get(
            "VLLM_DISABLE_COMPILE_CACHE", "<unset>"
        ),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "<unset>"),
        "ASCEND_RT_VISIBLE_DEVICES": os.environ.get(
            "ASCEND_RT_VISIBLE_DEVICES", "<unset>"
        ),
        "compilation_config": COMPILATION_CONFIG,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": GPU_MEM_UTIL,
    }


def trace_counts(log_text: str) -> dict:
    """Grep the per-leg log for plugin traces (ON proof / OFF zero-trace)."""
    return {
        "active_lines": log_text.count("fi_gelu activation path is ACTIVE"),
        "activation_path_lines": log_text.count("fi_gelu activation path"),
        "fi_gelu_mentions": log_text.count("fi_gelu"),
        "plugin_warn_lines": log_text.count("VLLM_HUST_FI_GELU=1 but"),
        "kernel_tag_lines": log_text.count("[fi_gelu]"),
    }


def parse_leg_json(log_text: str) -> dict | None:
    row = None
    for line in log_text.splitlines():
        if line.startswith(LEG_JSON_SENTINEL):
            try:
                row = json.loads(line[len(LEG_JSON_SENTINEL) :])
            except json.JSONDecodeError:
                continue
    return row


# --------------------------------------------------------------------------
# child: one leg = one process
# --------------------------------------------------------------------------
def run_leg(mode: str, leg_tag: str, port: int) -> int:
    if mode == "on":
        os.environ["VLLM_HUST_FI_GELU"] = "1"
    else:
        os.environ.pop("VLLM_HUST_FI_GELU", None)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    print(
        f"[leg {leg_tag}] begin mode={mode} port_slot={port} "
        f"VLLM_HUST_FI_GELU={os.environ.get('VLLM_HUST_FI_GELU', '<unset>')} "
        f"pid={os.getpid()}",
        flush=True,
    )

    t0 = time.monotonic()
    from vllm import LLM, SamplingParams  # noqa: PLC0415 -- env set above

    import_s = time.monotonic() - t0
    print(f"[leg {leg_tag}] `from vllm import LLM` {import_s:.1f}s", flush=True)

    prompts = build_prompts()
    t0 = time.monotonic()
    llm = LLM(
        model=MODEL,
        enable_prefix_caching=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_MEM_UTIL,
        tensor_parallel_size=1,
        enable_chunked_prefill=False,
        async_scheduling=False,
        compilation_config=COMPILATION_CONFIG,
    )
    engine_init_s = time.monotonic() - t0
    ready_s = time.monotonic() - T_PROC_START
    print(
        f"[leg {leg_tag}] LLM ready: engine_init={engine_init_s:.1f}s "
        f"ready_from_proc_start={ready_s:.1f}s",
        flush=True,
    )

    tokenizer = llm.get_tokenizer()
    prompt_lens = [len(tokenizer(p).input_ids) for p in prompts]

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall_s = time.perf_counter() - t0

    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    assert out_tokens == EXPECTED_OUT_TOKENS, (
        f"token total {out_tokens} != {EXPECTED_OUT_TOKENS}"
    )

    row = {
        "leg": leg_tag,
        "mode": mode,
        "ok": True,
        "port_slot": port,
        "wall_s": round(wall_s, 3),
        "ready_s": round(ready_s, 1),
        "import_s": round(import_s, 1),
        "engine_init_s": round(engine_init_s, 1),
        "out_tokens": out_tokens,
        "expected_out_tokens": EXPECTED_OUT_TOKENS,
        "n_prompts": N_PROMPTS,
        "max_tokens": MAX_TOKENS,
        "prompt_token_min": min(prompt_lens),
        "prompt_token_max": max(prompt_lens),
        "prompt_token_mean": round(statistics.fmean(prompt_lens), 1),
        "env": env_fingerprint(),
        "versions": version_triple(),
        "git": git_probe(REPO),
    }
    print(
        f"[leg {leg_tag}] WALL {wall_s:.3f}s out_tokens={out_tokens} "
        f"prompt_tok={min(prompt_lens)}-{max(prompt_lens)}",
        flush=True,
    )
    print(LEG_JSON_SENTINEL + json.dumps(row), flush=True)
    return 0


# --------------------------------------------------------------------------
# parent: orchestrator
# --------------------------------------------------------------------------
def write_results(rows: list[dict], summary: dict | None) -> None:
    payload = {
        "experiment": "W3.1 fi_gelu e2e A/B",
        "preregistration": str((RESULTS / "EXPECTATIONS-w31-e2e.md").relative_to(REPO)),
        "fixture": {
            "kind": "offline LLM.generate, one process per leg",
            "n_prompts": N_PROMPTS,
            "max_tokens": MAX_TOKENS,
            "expected_out_tokens": EXPECTED_OUT_TOKENS,
            "model": MODEL,
            "leg_order": [f"{m}_{t}" for m, t in LEGS],
            "port_slots": [PORT_BASE + i for i in range(len(LEGS))],
            "compilation_config": COMPILATION_CONFIG,
        },
        "legs": rows,
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")


def summarize(rows: list[dict]) -> dict:
    walls: dict[str, list[float]] = {"off": [], "on": []}
    for row in rows:
        if row.get("ok") and row.get("wall_s") is not None:
            walls[row["mode"]].append(row["wall_s"])

    out: dict = {
        "n_legs": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "rule": (
            "Δmedian = (median_off - median_on) / median_off; "
            "spread = max - min within an arm (normalized by that arm's median; "
            "the pre-registration writes 'spread = max-min' -- reported in both "
            "fractional and absolute-seconds form, the fractional form is the "
            "primary application of the rule, matching the W2b percent口径)"
        ),
    }
    for mode in ("off", "on"):
        w = sorted(walls[mode])
        if not w:
            continue
        med = statistics.median(w)
        out[f"{mode}_walls_s"] = [round(x, 3) for x in w]
        out[f"{mode}_median_s"] = round(med, 3)
        out[f"{mode}_min_s"] = round(min(w), 3)
        out[f"{mode}_max_s"] = round(max(w), 3)
        out[f"{mode}_spread_s"] = round(max(w) - min(w), 3)
        out[f"{mode}_spread_frac"] = round((max(w) - min(w)) / med, 5)

    if "off_median_s" in out and "on_median_s" in out:
        off_med = out["off_median_s"]
        on_med = out["on_median_s"]
        delta = (off_med - on_med) / off_med
        band_frac = max(out["off_spread_frac"], out["on_spread_frac"])
        band_s = max(out["off_spread_s"], out["on_spread_s"])
        out["delta_median"] = round(delta, 5)
        out["delta_median_pct"] = round(100.0 * delta, 3)
        out["delta_median_abs_s"] = round(off_med - on_med, 3)
        out["noise_band_frac"] = band_frac
        out["noise_band_s"] = band_s
        out["within_noise_band_frac"] = bool(abs(delta) < band_frac)
        out["within_noise_band_abs_s"] = bool(abs(off_med - on_med) < band_s)
        if out["within_noise_band_frac"]:
            out["verdict"] = (
                "NOISE-BAND / NOT DECIDABLE (|Δmedian| < max(spread_off, "
                "spread_on), normalized spread)"
            )
        else:
            out["verdict"] = (
                "WIN (ON faster) " if delta > 0 else "LOSS (ON slower) "
            ) + f"Δmedian={100.0 * delta:+.3f}%"
        pct = abs(out["delta_median_pct"])
        if pct < 0.2:
            consistency = (
                "below projected band (0.2-0.6%; projection says low end is expected)"
            )
        elif pct <= 0.6:
            consistency = "within projected band (0.2-0.6%)"
        else:
            consistency = "above projected band (0.2-0.6%)"
        out["projection"] = {
            "projected_abs_delta_pct": [0.2, 0.6],
            "source": "EXPECTATIONS-w31-e2e.md (graph replay 0.50ms/step @ TPOT~112ms)",
            "consistency": consistency,
        }
    return out


def print_summary(rows: list[dict], summary: dict) -> None:
    print("\n" + "=" * 78, flush=True)
    print("W3.1 fi_gelu e2e A/B -- summary", flush=True)
    print("=" * 78, flush=True)
    print(
        f"{'leg':<8}{'mode':<6}{'wall_s':>10}{'ready_s':>10}{'rc':>5}"
        f"{'ok':>4}{'ACTIVE':>8}{'fi_gelu':>9}",
        flush=True,
    )
    for row in rows:
        tr = row.get("traces", {})
        print(
            f"{row['leg']:<8}{row['mode']:<6}"
            f"{row.get('wall_s', float('nan')):>10}"
            f"{row.get('ready_s', float('nan')):>10}"
            f"{str(row.get('subprocess_rc', '')):>5}"
            f"{str(row.get('ok')):>4}"
            f"{tr.get('active_lines', 0):>8}"
            f"{tr.get('fi_gelu_mentions', 0):>9}",
            flush=True,
        )
    print("-" * 78, flush=True)
    for mode in ("off", "on"):
        if f"{mode}_median_s" in summary:
            print(
                f"{mode.upper():<4} walls={summary[f'{mode}_walls_s']} "
                f"median={summary[f'{mode}_median_s']}s "
                f"spread={summary[f'{mode}_spread_s']}s "
                f"({100 * summary[f'{mode}_spread_frac']:.2f}%)",
                flush=True,
            )
    if "delta_median" in summary:
        print(
            f"Δmedian={summary['delta_median_pct']:+.3f}% "
            f"({summary['delta_median_abs_s']:+.3f}s)  "
            f"noise_band(frac)={summary['noise_band_frac']:.5f} "
            f"noise_band(s)={summary['noise_band_s']:.3f}",
            flush=True,
        )
        print(f"VERDICT: {summary['verdict']}", flush=True)
        print(f"projection: {summary['projection']['consistency']}", flush=True)
    print("=" * 78 + "\n", flush=True)


def write_summary_txt(rows: list[dict], summary: dict) -> None:
    lines = []
    lines.append("W3.1 fi_gelu e2e A/B -- mechanical summary")
    lines.append("preregistration: bench/results/EXPECTATIONS-w31-e2e.md")
    lines.append("")
    lines.append("per-leg")
    hdr = (
        "leg",
        "mode",
        "wall_s",
        "ready_s",
        "import_s",
        "engine_init_s",
        "out_tokens",
        "rc",
        "ok",
        "ACTIVE",
        "fi_gelu",
    )
    lines.append("\t".join(hdr))
    for row in rows:
        tr = row.get("traces", {})
        lines.append(
            "\t".join(
                str(x)
                for x in (
                    row["leg"],
                    row["mode"],
                    row.get("wall_s"),
                    row.get("ready_s"),
                    row.get("import_s"),
                    row.get("engine_init_s"),
                    row.get("out_tokens"),
                    row.get("subprocess_rc"),
                    row.get("ok"),
                    tr.get("active_lines"),
                    tr.get("fi_gelu_mentions"),
                )
            )
        )
    lines.append("")
    lines.append("summary")
    for key, val in summary.items():
        lines.append(f"{key} = {val}")
    lines.append("")
    lines.append(
        "note: offline LLM fixture (EXPECTATIONS) binds no HTTP port; the "
        "8341+ slot per leg is a bookkeeping guard checked free before/after."
    )
    OUT_TXT.write_text("\n".join(lines) + "\n")


def orchestrate() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    self_path = str(Path(__file__).resolve())
    print(
        f"[orch] W3.1 gelu e2e A/B: {len(LEGS)} legs -> {OUT_JSON}\n"
        f"[orch] card={os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>')} "
        f"cwd={os.getcwd()} logs={LOG_DIR}",
        flush=True,
    )

    rows: list[dict] = []
    for idx, (mode, tag) in enumerate(LEGS):
        leg_tag = f"{mode}_{tag}"
        port = PORT_BASE + idx
        log_path = LOG_DIR / f"leg_{leg_tag}.log"

        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["VLLM_DISABLE_COMPILE_CACHE"] = "1"
        env.pop("VLLM_HUST_FI_GELU", None)
        if mode == "on":
            env["VLLM_HUST_FI_GELU"] = "1"

        free_before = port_is_free(port)
        print(
            f"[orch] leg {idx + 1}/{len(LEGS)} {leg_tag} port_slot={port} "
            f"free_before={free_before} start={time.strftime('%H:%M:%S')}",
            flush=True,
        )

        rc: object = -1
        with open(log_path, "w") as fh:
            proc = subprocess.Popen(
                [sys.executable, self_path, "--leg", mode, leg_tag, str(port)],
                stdout=fh,
                stderr=subprocess.STDOUT,
                cwd="/tmp",
                env=env,
            )
            try:
                rc = proc.wait(timeout=LEG_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                rc = "timeout"

        free_after = port_is_free(port)
        text = log_path.read_text(errors="replace")
        row = parse_leg_json(text) or {
            "leg": leg_tag,
            "mode": mode,
            "wall_s": None,
            "ok": False,
        }
        row["index"] = idx + 1
        row["leg"] = leg_tag
        row["mode"] = mode
        row["port_slot"] = port
        row["port_slot_free_before"] = free_before
        row["port_slot_free_after"] = free_after
        row["subprocess_rc"] = rc
        row["ok"] = bool(row.get("ok")) and rc == 0
        row["log"] = str(log_path.relative_to(REPO))
        row["traces"] = trace_counts(text)
        rows.append(row)

        print(
            f"[orch] leg {idx + 1}/{len(LEGS)} {leg_tag} done rc={rc} "
            f"wall={row.get('wall_s')} free_after={free_after} "
            f"ACTIVE={row['traces']['active_lines']} "
            f"fi_gelu={row['traces']['fi_gelu_mentions']}",
            flush=True,
        )
        write_results(rows, None)  # incremental: survive an aborted run

        if idx < len(LEGS) - 1:
            print(f"[orch] sleep {LEG_SLEEP_S}s (thermal convergence)", flush=True)
            time.sleep(LEG_SLEEP_S)

    summary = summarize(rows)
    write_results(rows, summary)
    print_summary(rows, summary)
    write_summary_txt(rows, summary)
    print(f"[orch] DONE -> {OUT_JSON} / {OUT_TXT}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="W3.1 fi_gelu e2e A/B harness")
    parser.add_argument("--leg", action="store_true", help="run a single child leg")
    parser.add_argument("mode", nargs="?", choices=["off", "on"], help="child arm")
    parser.add_argument("tag", nargs="?", help="child leg tag")
    parser.add_argument("port", nargs="?", type=int, help="child port slot")
    args = parser.parse_args()
    if args.leg:
        if not (args.mode and args.tag and args.port):
            parser.error("--leg requires: <mode> <tag> <port>")
        return run_leg(args.mode, args.tag, args.port)
    return orchestrate()


if __name__ == "__main__":
    sys.exit(main())
