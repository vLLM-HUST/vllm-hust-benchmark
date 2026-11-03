# Interpreting cache counters under the 2 GiB reservation

The 8 GiB service experiment requires zero cached prompt tokens for every synthetic request. The
separate 2 GiB pressure experiment retains cached-token counters and requires their review against
unique request salts and preemption evidence. The original pressure protocol already made this
distinction. `pressure-policy-amendment.json`, both frozen contracts and the public preregistration
retain the correction made before any current-release 2 GiB requests. The amendment changes client
validation metadata only; request payloads, salts, order and output budgets stay identical between
the service budgets.

## What the pinned host implements

All links below refer to the measured vLLM revision `d0f22d2bda562156e4dbf433ce645e1769b4f804`.

1. The completion renderer passes `cache_salt` into prompt extras. The engine input and request
   preserve it. See
   [completion rendering](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/renderers/online_renderer.py#L302),
   [input processing](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/engine/input_processor.py#L380)
   and
   [request state](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/request.py#L154).
1. The salt participates in the first block's extra hash keys. Subsequent hashes include the parent
   hash, so distinct salts separate complete prefix chains, subject to the hash implementation's
   normal collision assumptions. See
   [cache keys and block hashing](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/core/kv_cache_utils.py#L554).
1. Preemption frees request blocks, resets `num_computed_tokens` to zero and requeues the request.
   Scheduling a request with zero computed tokens performs prefix lookup again. This permits reuse
   of that same request's cached prefix after preemption. See
   [preemption](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/core/sched/scheduler.py#L1258)
   and
   [prefix lookup](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/core/sched/scheduler.py#L710).
1. `PrefillStats.set` updates cached-token counts while the stats object remains present. The object
   is consumed when the engine emits output; partial prefill emits no output. Preemption during
   initial prefill can therefore affect the cache count reported with the first output. See
   [prefill statistics](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/metrics/stats.py#L246),
   [statistics update](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/core/sched/scheduler.py#L804),
   [output and partial prefill](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/core/sched/scheduler.py#L1880)
   and
   [usage propagation](https://github.com/vLLM-HUST/vllm-hust/blob/d0f22d2bda562156e4dbf433ce645e1769b4f804/vllm/v1/engine/output_processor.py#L628).

## Evidence and limits

The offline checks join every recorded service request with its frozen salt, HTTP completion, engine
completion count and group. A cached service request must have a distinct salt and belong to a phase
whose sampled cumulative preemption count increases. The report retains every cached case and phase
counter difference, including excluded warmups. It also retains every repeated compression commit
and both TP acknowledgements if recomputation causes any.

This is a source-based attribution supported by unique salts and phase counters. The retained INFO
logs do not contain an individual preemption event trace for each request. Phase counter differences
are sampled observations; they are not exact attribution of every preemption to a specific request.
The matched pressure result measures the complete serving behavior, including preemption and any
same-request prefix reuse. It must not be described as an all-uncached experiment or as an isolated
compression-kernel speedup.

The capacity probes have their own frozen inputs and no service-style per-request salts. Report
their cached counts separately; do not transfer the service isolation argument to them. C1/C2/C4
completions at configured reservations establish tested operating points, not maximum resident
capacity. KV block ownership changes likewise do not establish a reduction in physical HBM use.
