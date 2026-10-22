"""Serving-level (HTTP) cascade OFF vs ON exact-match comparison.

Starts `vllm serve` twice on the same card/port config (OFF: cascade env unset;
ON: cascade env injected), sends the SAME prompts with temperature=0 and
ignore_eos so the decode phase is not diluted, and compares the returned text
per prompt (exact match + first-divergence character position).

Usage: ASCEND_RT_VISIBLE_DEVICES=6 python ev2_serve_compare.py
Writes /tmp/ev2_serve_{off,on}.json and prints the comparison.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

MODEL = "/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct"
PORT = 8351
N_SENT = 420
BATCH = 8
GEN = 128
LOG_DIR = "/vllm-workspace/flashinfer-migration/cascade-evidence/logs/v1-ev2"

base = "You are a helpful assistant. " + (
    "The quick brown fox jumps over the lazy dog. " * N_SENT
)
PROMPTS = [
    base + f"\n\nQuestion {i}: What is {i}+{i}? Answer with the number only."
    for i in range(BATCH)
]


def wait_ready(log_path, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(log_path):
            txt = open(log_path, errors="replace").read()
            if "Application startup complete" in txt:
                return True
        if proc.poll() is not None:
            return False
        time.sleep(5)
    return False


def run_leg(mode):
    log = f"{LOG_DIR}/serve_{mode}_card6.log"
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["ASCEND_RT_VISIBLE_DEVICES"] = "6"
    if mode == "on":
        env["VLLM_ASCEND_ENABLE_CASCADE_DECODE"] = "1"
        env["VLLM_ASCEND_ENABLE_CASCADE_GRAPH"] = "0"  # graph lane crashes (see report)
        env["VLLM_ASCEND_CASCADE_MIN_PREFIX"] = "4096"
        env["VLLM_ASCEND_CASCADE_MIN_REQS"] = "2"
    with open(log, "w") as fh:
        proc = subprocess.Popen(
            [
                "vllm", "serve", MODEL,
                "--max-model-len", "8192",
                "--enforce-eager",
                "--port", str(PORT),
                "--gpu-memory-utilization", "0.85",
            ],
            stdout=fh, stderr=subprocess.STDOUT, env=env, cwd="/tmp",
        )
    if not wait_ready(log, proc):
        proc.kill()
        raise RuntimeError(f"serve {mode} did not become ready; see {log}")
    # Fire all prompts CONCURRENTLY so the server schedules them as one
    # shared-prefix batch (the cascade gate needs >= MIN_REQS requests in a
    # step; a serial loop gives B=1 and never opens the gate).
    outs = [None] * len(PROMPTS)
    errs = []

    def one(i, p):
        try:
            body = json.dumps(
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": p}],
                    "temperature": 0,
                    "max_tokens": GEN,
                    "ignore_eos": True,
                }
            ).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{PORT}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=900) as r:
                data = json.loads(r.read())
            outs[i] = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            errs.append((i, repr(exc)))

    threads = [
        threading.Thread(target=one, args=(i, p)) for i, p in enumerate(PROMPTS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        proc.terminate()
        raise RuntimeError(f"serve {mode} request errors: {errs}")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    return outs


def main():
    off = run_leg("off")
    on = run_leg("on")
    json.dump({"off": off, "on": on}, open("/tmp/ev2_serve_compare.json", "w"))
    div = []
    for i, (a, b) in enumerate(zip(off, on)):
        if a != b:
            fd = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            div.append((i, fd, len(a), len(b)))
    print(f"SERVE-COMPARE prompts={len(off)} exact_match={len(off)-len(div)} "
          f"diverged={len(div)}")
    for d in div:
        print(f"  prompt {d[0]}: first_diff_char={d[1]} lenOFF={d[2]} lenON={d[3]}")
    print("SERVE-COMPARE DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
