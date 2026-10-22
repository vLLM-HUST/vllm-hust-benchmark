"""ev3 sweep harness — frozen upstream harness + two extra legs.

Frozen upstream harness: cascade_sweep_plug.py.frozen (sha256
3dbcd45630ac3dabece70f36f47cfeefa74fd2eca72696bb55d3912c1a852557, identical to
/vllm-workspace/cascade-e2e/cascade_sweep_plug.py at capture time).  The
upstream file only knows mode in {off, on} (on == graph mode: decode+graph env).
This copy adds:

  on_eager : VLLM_ASCEND_ENABLE_CASCADE_DECODE=1 only, graph env UNSET, and
             enforce_eager=True — the eager two-stage lane.  This leg engages
             cascade WITHOUT the graph-capture monkeypatch, so it is unaffected
             by the host `_capture_cudagraphs(profiler=...)` signature drift.
  off_eager: graph env unset, enforce_eager=True (the matching OFF control).

Everything else (model, prompts, sampling params, shapes, graph config) is
byte-identical to the frozen harness.

Usage: [BATCH=n] [ONLY=p420_b64] python cascade_sweep_ev3.py MODE TAG
"""
import json
import os
import sys
import time

mode, tag = sys.argv[1], sys.argv[2]
if mode in ("on", "on_eager"):
    os.environ["VLLM_ASCEND_ENABLE_CASCADE_DECODE"] = "1"
    os.environ["VLLM_ASCEND_CASCADE_MIN_PREFIX"] = "4096"
    os.environ["VLLM_ASCEND_CASCADE_MIN_REQS"] = "2"
if mode == "on":
    os.environ["VLLM_ASCEND_ENABLE_CASCADE_GRAPH"] = "1"

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "/data/shared_models/Qwen--Qwen2.5-Coder-14B-Instruct"
BASE = "You are a helpful assistant. "
SENT = "The quick brown fox jumps over the lazy dog. "
SHAPES = [(420, 32), (420, 64), (420, 128),      # ~4.4k shared prefix
          (800, 32), (800, 64), (800, 128),      # ~8.2k
          (1600, 32), (1600, 64), (1600, 128)]   # ~16.3k

llm_kwargs = dict(
    model=MODEL,
    enable_prefix_caching=True,
    max_model_len=20480,
    max_num_batched_tokens=32768,
    gpu_memory_utilization=0.85,
    tensor_parallel_size=1,
    cudagraph_capture_sizes=[32, 64, 128],
    enable_chunked_prefill=False,
    async_scheduling=False,
)
if mode.endswith("eager"):
    llm_kwargs["enforce_eager"] = True
llm = LLM(**llm_kwargs)

# ignore_eos is MANDATORY (same discipline as the frozen harness): without it
# generation ends after ~8 tokens and the decode-phase comparison is diluted.
sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)

only = os.environ.get("ONLY", "")
batch_filter = os.environ.get("BATCH", "")

results = {}
for n_rep, batch in SHAPES:
    key = f"p{n_rep}_b{batch}"
    if only and key != only:
        continue
    if batch_filter and str(batch) != batch_filter:
        continue
    base = BASE + SENT * n_rep
    prompts = [base + f"\n\nQuestion {i}: What is {i}+{i}? Answer with the number only."
               for i in range(batch)]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    toks = [list(o.outputs[0].token_ids) for o in outs]
    results[key] = toks
    json.dump(results, open(f"/tmp/sweep_{mode}_{tag}.json", "w"))
    print(f"SWEEP mode={mode} tag={tag} {key} wall={wall:.2f}s "
          f"out_tokens={sum(len(t) for t in toks)}", flush=True)

print(f"SWEEP mode={mode} tag={tag} ALL DONE", flush=True)
