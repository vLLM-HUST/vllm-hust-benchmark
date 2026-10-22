"""Evidence #2 (correctness alignment) harness on the v1 baseline.

Faithful port of the historical C3 acceptance harness
(cascade-c3-results/probes/c3_run_14b_graph.py + cascade-e2e/cascade_c3_plug.py),
parameterized by env so the same script covers the graph and eager lanes.

Usage (CWD must NOT be /vllm-workspace):
  ASCEND_RT_VISIBLE_DEVICES=6 python ev2_run.py MODE TAG

MODE in {off, on, on_run2, on_fp32, on_torchmerge}.
Env knobs:
  EV2_SENT=420      repeated sentences in the shared prefix (~4356 tok at 420)
  EV2_BATCH=64      number of prompts
  EV2_GEN=128       max_tokens
  EV2_MAXLEN=8192   max_model_len
  EV2_EAGER=0       1 -> enforce_eager (no aclgraph)
  EV2_CAPTURE=32,64 cudagraph_capture_sizes (graph lane)
  EV2_MIN_PREFIX=4096 / EV2_MIN_REQS=2  cascade gate overrides
  EV2_TORCHMERGE=1  force the torch (non-kernel) merge fallback
  EV2_IGNORE_EOS=0  1 -> ignore_eos (full 128-step decode walls)
  EV2_SHIM_SIG=1    DIAGNOSTIC ONLY: install a harness-side signature shim that
                    adapts the plugin's stale host-method calls to the v1 host
                    signatures (NOT an acceptance result -- see report).
Dump -> /tmp/ev2_{MODE}_{TAG}.json (list of token-id lists, one per prompt).
"""
import json
import os
import sys
import time

mode, tag = sys.argv[1], sys.argv[2]
if mode.startswith("on"):
    os.environ["VLLM_ASCEND_ENABLE_CASCADE_DECODE"] = "1"
    os.environ["VLLM_ASCEND_CASCADE_MIN_PREFIX"] = os.environ.get(
        "EV2_MIN_PREFIX", "4096"
    )
    os.environ["VLLM_ASCEND_CASCADE_MIN_REQS"] = os.environ.get("EV2_MIN_REQS", "2")
    os.environ["VLLM_ASCEND_ENABLE_CASCADE_GRAPH"] = os.environ.get("EV2_GRAPH", "1")
    if mode == "on_fp32":
        os.environ["VLLM_ASCEND_CASCADE_PRECISION"] = "fp32"

if os.environ.get("EV2_SHIM_SIG") == "1":
    # DIAGNOSTIC ONLY (see ev2_shim.py / report).  Must run BEFORE `from vllm
    # import LLM`, because it wraps the plugin install() functions that
    # vllm.general_plugins invokes during import.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ev2_shim

    ev2_shim.install()

if os.environ.get("EV2_SKIP_GRAPH_INSTALL") == "1" or os.environ.get(
    "EV2_SKIP_RUNNER_PATCH"
) == "1":
    # DIAGNOSTIC ONLY: neutralize one of the two graph-plugin install steps
    # before vllm.general_plugins runs load(), to attribute a failure.
    import vllm_ascend_split_batch.cascade_graph_plugin as _gp
    import vllm_ascend_split_batch.cascade_runner_patch as _rp

    if os.environ.get("EV2_SKIP_GRAPH_INSTALL") == "1":
        _gp.install = lambda *a, **k: True
        print("[ev2] DIAGNOSTIC: cascade_graph_plugin.install neutralized", flush=True)
    if os.environ.get("EV2_SKIP_RUNNER_PATCH") == "1":
        _rp.install = lambda *a, **k: True
        print("[ev2] DIAGNOSTIC: cascade_runner_patch.install neutralized", flush=True)

from vllm import LLM, SamplingParams  # noqa: E402

# sole deviation from the verbatim harness (cascade/ev2_run.py): model path
# -> the REAL Qwen2.5-14B-Instruct (stand-in was Qwen--Qwen2.5-Coder-14B-Instruct).
MODEL = "/data/shared_models/Qwen--Qwen2.5-14B-Instruct"
N_SENT = int(os.environ.get("EV2_SENT", "420"))
BATCH = int(os.environ.get("EV2_BATCH", "64"))
GEN = int(os.environ.get("EV2_GEN", "128"))
MAXLEN = int(os.environ.get("EV2_MAXLEN", "8192"))
EAGER = os.environ.get("EV2_EAGER", "0") == "1"
CAPTURE = [int(x) for x in os.environ.get("EV2_CAPTURE", "32,64").split(",")]
IGNORE_EOS = os.environ.get("EV2_IGNORE_EOS", "0") == "1"

base = "You are a helpful assistant. " + (
    "The quick brown fox jumps over the lazy dog. " * N_SENT
)
prompts = [
    base + f"\n\nQuestion {i}: What is {i}+{i}? Answer with the number only."
    for i in range(BATCH)
]

kwargs = dict(
    model=MODEL,
    enable_prefix_caching=True,
    max_model_len=MAXLEN,
    gpu_memory_utilization=0.85,
    tensor_parallel_size=1,
    enable_chunked_prefill=False,
    async_scheduling=False,
)
if EAGER:
    kwargs["enforce_eager"] = True
else:
    kwargs["cudagraph_capture_sizes"] = CAPTURE

llm = LLM(**kwargs)

if os.environ.get("EV2_TORCHMERGE") == "1":
    # Force the plugin's torch (non-kernel) merge fallback: hide the kernel op
    # so `_HAS_LSE_MERGE_OP`-style probes miss it at the call site.
    import vllm_ascend_split_batch.cascade_plugin as _cp

    _cp._HAS_LSE_MERGE_OP = False
    print("[ev2] forced torch-merge fallback (_HAS_LSE_MERGE_OP=False)", flush=True)

sp = SamplingParams(max_tokens=GEN, temperature=0.0, ignore_eos=IGNORE_EOS)
t0 = time.perf_counter()
outs = llm.generate(prompts, sp, use_tqdm=False)
wall = time.perf_counter() - t0
toks = [list(o.outputs[0].token_ids) for o in outs]
path = f"/tmp/ev2_{mode}_{tag}.json"
json.dump(toks, open(path, "w"))
print(
    f"EV2 mode={mode} tag={tag} batch={BATCH} sent={N_SENT} gen={GEN} "
    f"eager={int(EAGER)} wall={wall:.2f}s "
    f"out_tokens={sum(len(t) for t in toks)} -> {path} DONE",
    flush=True,
)
