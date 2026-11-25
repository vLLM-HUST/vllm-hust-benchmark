"""Drive `vllm bench serve` as a CLIENT against a running server and hash its inputs and outputs.

`vllm bench serve` only needs the vllm package on the client machine; it sends requests to --base-url. This
module starts no server. Flags follow the B0 runs: openai-chat, custom dataset, no shuffle, request-rate inf,
temperature 0, seed 0.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

from .hashing import sha256_file


def build_command(vllm_bin: str, base_url: str, served_model: str, tokenizer: str, dataset: str,
                  num_prompts: int, output_len: int, result_dir: str, result_name: str) -> List[str]:
    return [vllm_bin, "bench", "serve", "--backend", "openai-chat",
            "--base-url", base_url.rstrip("/"), "--endpoint", "/v1/chat/completions",
            "--model", served_model, "--tokenizer", tokenizer,
            "--dataset-name", "custom", "--dataset-path", dataset, "--disable-shuffle",
            "--custom-output-len", str(output_len), "--num-prompts", str(num_prompts), "--request-rate", "inf",
            "--temperature", "0", "--seed", "0",
            "--save-result", "--result-dir", result_dir, "--result-filename", result_name + ".json"]


def bench(run_dir, name: str, dataset: str, base_url: str, served_model: str, tokenizer: str,
          num_prompts: int = 200, output_len: int = 256, vllm_bin: str = "vllm", dry_run: bool = False) -> dict:
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('name must be a filename stem')
    if num_prompts <= 0 or output_len <= 0:
        raise ValueError('positive prompt count and output length required')
    run = Path(run_dir)
    outputs, inputs = run / "outputs", run / "inputs"
    result = outputs / (name + ".json")
    if result.exists():
        raise FileExistsError("%s exists; never overwrite - use a new run dir" % result)
    if not Path(dataset).is_file():
        raise FileNotFoundError(dataset)
    outputs.mkdir(parents=True, exist_ok=True)
    inputs.mkdir(parents=True, exist_ok=True)

    with open(dataset, 'rb') as source:
        available = sum(bool(line.strip()) for line in source)
    if num_prompts > available:
        raise ValueError('num-prompts exceeds dataset rows; do not silently repeat cases')
    import shutil
    frozen = inputs / (name + '.materialized.jsonl')
    if frozen.exists():
        raise FileExistsError('bench inputs already exist; use another run directory')
    shutil.copyfile(dataset, frozen)
    dataset = str(frozen)
    cmd = build_command(vllm_bin, base_url, served_model, tokenizer, dataset, num_prompts, output_len,
                        str(outputs), name)
    (outputs / (name + ".cmd.txt")).write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")
    (inputs / (name + ".materialized.sha256")).write_text("%s  %s\n" % (sha256_file(dataset), dataset))
    with open(dataset, "rb") as f:
        rows = sum(1 for _ in f)
    (inputs / (name + ".rows.txt")).write_text("%d\n" % rows)
    if dry_run:
        return {"command": cmd, "dataset_rows": rows}

    with open(outputs / (name + ".bench.log"), "wb") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not result.exists():
        raise RuntimeError("vllm bench serve failed (exit %s); see %s" % (proc.returncode, outputs / (name + ".bench.log")))
    data = json.loads(result.read_text())
    keys = ["date", "num_prompts", "completed", "failed", "duration", "request_throughput", "output_throughput",
            "total_token_throughput", "mean_ttft_ms", "mean_tpot_ms"]
    summary = {k: data.get(k) for k in keys}
    (outputs / (name + ".summary.json")).write_text(json.dumps(summary, indent=1))
    (outputs / (name + ".json.sha256")).write_text("%s  %s\n" % (sha256_file(result), result.name))
    return summary
