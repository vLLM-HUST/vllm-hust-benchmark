# Compression Method Architecture

English | [简体中文](methods.zh.md)

## Runtime layers

Since version 0.6, this is a self-contained plugin and does not use the removed
vLLM-HUST compression lifecycle:

1. `plugin.py` is the opt-in `vllm.general_plugins` entry point. It patches
   scheduler-side symbols immediately and installs the Ascend worker hook
   lazily, so manager and API-only processes do not import the NPU runner.
2. `calibration.py` generates a missing model-bound artifact before serving
   weights load, or exposes the same generator through a console command.
3. `stateful.py` owns scheduler/KV-manager state. It commits a completed
   compression at the next synchronous scheduling barrier and frees the old
   block-table tail.
4. `provider.py` validates current host objects, mirrors transactions in the
   worker, translates semantic positions to physical slots, and delegates
   selection/materialization to a method.
5. `methods/<name>/` owns an algorithm's options, compatibility checks,
   scoring, calibration, and KV materialization.

The current adapter deliberately depends on the internal host symbols listed
in the main README. Those interfaces are not frozen, so support must remain on
the tested host version line and fail closed after incompatible changes.

## Configuration contract

The manager stores one JSON object. `schema_version`, `provider`, `method`, and
`method_config` are required. For example:

```json
{
  "schema_version": 1,
  "provider": "ascend_kvcompress",
  "method": "triattention",
  "method_config": {
    "stats_path": "/absolute/path/stats.pt",
    "auto_calibrate": true,
    "calibration_input_path": "/absolute/path/calibration.txt",
    "calibration_max_length": 4096,
    "kv_budget": 8192,
    "recompute_window": 1024,
    "position_policy": "v3",
    "protected_prefix_window": 128,
    "protected_recent_window": 512,
    "position_segments": 8,
    "score_aggregation": "mean",
    "layer_aggregation": "mean",
    "score_chunk_size": 8192,
    "score_layer_stride": 8,
    "min_output_tokens_for_compression": 64
  }
}
```

Unknown keys, methods, incompatible block alignment, and invalid calibration
files are rejected rather than ignored. A missing file is generated only when
`auto_calibrate` is enabled; see the calibration artifact guide.

`min_output_tokens_for_compression` is a non-negative requested-output gate.
When a request's maximum generation is below it, both scheduler and worker skip
the transaction. `0` preserves the legacy always-eligible behavior; `64` is the
quality-qualified public-benchmark default that removes overhead from the
32-token retrieval workload.

The standard runtime has equal 128-token scheduler, attention-manager, and
kernel-cache blocks. For the explicitly supported Qwen3.5 hybrid, Ascend may
promote the full-attention manager page to 1,152 or 2,048 tokens while keeping
128-token kernel blocks. The adapter validates scheduler alignment against
the LCM of every cache manager, then expands each attention page into 9 or 16
kernel blocks. The method's maximum physical token count must be divisible by
the active attention page, not by the cross-group alignment.

`score_layer_stride=8` uniformly samples six of the 48 calibrated layers for
selection while all 48 K/V layers are still materialized. On the frozen public
Qasper and LongBench-v2 sets this setting preserved baseline quality and
improved the matched end-to-end result compared with the former value `4`.

## V3 position policy

`position_policy=global` retains the legacy global top-k behavior. The opt-in
`v3` policy implements the paper's three-part position treatment:

1. `protected_prefix_window` tokens at the start are always retained.
2. `protected_recent_window` tokens at the end are always retained.
3. The middle context is split into `position_segments` equal ranges. The
   exact global eviction count is distributed proportionally across those
   ranges, then each range keeps its own highest-scoring tokens.

The implementation uses integer cumulative quotas, so it always selects
exactly `kv_budget` positions even when segment lengths are uneven. For
equal-width middle segments on an NPU, it batches all segment top-k operations
into one call and caches the small, shape-dependent index plan; uneven or
zero-quota shapes retain the original per-segment path. This changes execution
cost, not the quota rule or protected positions.

## Qwen3.5 hybrid path

Qwen3.5-35B-A3B has 40 text layers arranged as ten repetitions of three
Gated-DeltaNet layers and one full-attention layer. The method binds only full
attention layers 3, 7, …, 39 and leaves recurrent-state groups unchanged. Its
256-dimensional heads use partial RoPE over only 64 dimensions. Calibration
therefore stores `q_pass_mean` for the remaining 192 dimensions. The plugin's
fused paged-key scorer now adds this direct content dot product to the rotary
trigonometric score in one NPU kernel; its Torch fallback remains available for
non-mean aggregation. The fused partial-RoPE path passed numerical checks on
an Ascend NPU. A subsequent official SWE pair narrowed but did not reverse
the performance regression; see [Frontier benchmarking](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/aa041953616fa170014a0c7f43fb06bf1c59b846/archive/mod-targeted-20261023/compress-history/snapshot/docs/archive/results/frontier-benchmarking-0.9.md).

With tensor parallelism, calibration rows are sharded by local KV head and an
all-reduce maximum synchronizes per-layer scores before selection. TP size must
divide the model's KV heads. Qwen3.5's `mamba_cache_mode=align` is supported
through the plugin's transaction-aware cache handling, alongside `none`;
other hybrid layouts and PP/DP/DCP/PCP remain rejected.

## Method contract

Implement `KVCompressionMethod` from `methods/base.py`:

- `name`: stable registry/configuration name.
- `runtime_spec`: block-aligned threshold, recompute window, maximum physical
  length, and destination requirements visible to the scheduler adapter.
- `compatibility_reasons(worker)`: side-effect-free method checks.
- `bind_model_runner(runner, layer_caches)`: allocate state after common cache
  validation.
- `compress(request)`: synchronously materialize one transaction and return
  the new physical length, plus optional per-layer lengths.

For requests eligible to cross the method's compression threshold, automatic
prefix-cache admission caps the reusable prefix at
`max(0, min(prompt_tokens - 1, prompt_tokens - required_recompute_tokens))`.
This guarantees that the method's declared trailing query suffix is executed,
including with chunked prefill. Prompts shorter than the required window
recompute from token zero. APC-disabled, below-threshold, and output-ineligible
requests keep the host's original admission path.
Query-observing methods must declare a recompute window at least as long as
`query_window_tokens`. With APC enabled, the adapter checks the required host
lookup hooks at startup and fails closed if they are unavailable.

### Per-layer physical state

`CompressionResult.per_layer_physical_num_tokens` may contain one `(layer_name,
length)` pair for every bound full-attention layer. Names must be unique and
exact; lengths must be positive integers no greater than
`runtime_spec.max_physical_num_tokens`. The scheduler reserves private blocks
using that declared group maximum. The worker switches the per-layer anchors
with the same acknowledged transaction and advances each layer's KV write
slots and attention lengths independently during decode. A later
`CompressionRequest.per_layer_physical_num_tokens` reports the current lengths
if the request is compressed again. GDN continues to receive semantic lengths.

`None` and all-equal-to-maximum maps use the established shared metadata path,
including TriAttention. A method that may return unequal maps must set
`requires_per_layer_physical_state` before model loading. Eager execution can
still discover an undeclared unequal result at transaction time; graph capture
cannot, because every layer needs a distinct address-stable slot buffer before
the first capture.

The declared path supports eager execution and the validated
`FULL_AND_PIECEWISE` Ascend FIA update seam. FULL replay rebinds each captured
attention task by layer name, while piecewise prefill continues through the
Python observation hooks. With qualified Qwen3.5 MTP2, the adapter also gives
the auxiliary draft-attention cache its own sequence lengths and slots through
the speculative common metadata. Rejected draft positions therefore overwrite
the same per-layer physical slots on the next step. Other graph modes,
attention backends, speculative methods, and context-parallel attention fail
closed. CPU contract tests do not qualify an external method for serving.

### Optional query observation

`query_window_tokens` defaults to `0`; existing methods, including
TriAttention, install no query hooks. A method that sets a positive integer
must implement `capture_query(layer, query, spans)`,
`complete_query_observation(observation)`, and
`discard_query_observation(request_id)`.

`query_layer_indices` defaults to `None`, meaning every bound full-attention
cache layer. A hybrid speculative runtime may bind an auxiliary draft cache
that does not execute in the target-model prefill. Query-based methods can
return the exact target-layer indices they consume; the adapter then observes
and completes only those layers while the auxiliary cache remains part of the
materialized per-layer result.

The host calls `capture_query` only for bound full-attention layers during
prefill. `query` is the attention layer's post-position-encoding query tensor;
each `QueryBatchSpan` gives one request's half-open row range in that tensor.
The method owns any retained tensors and should allocate address-stable buffers
in `bind_model_runner`; it can accumulate the trailing window across chunked
prefill steps. The host publishes no observation until the complete model
forward and sampling succeed. On the final prefill step with an authorized
compression plan, `complete_query_observation` receives the request ID, exact
`CompressionPlan`, semantic length, requested window length, and participating
layer indices. The same plan is then present on `CompressionRequest.plan`.
The method must check that every required layer has a complete window before
using it. The host calls `discard_query_observation` after compression and on
request finish, preemption, restart, or failed execution.

The host fails closed when an observing method misses a requested layer during
a prefill forward, including a graph path that bypasses the Python hook. This
CPU-tested contract does not qualify any graph mode or external method for
serving. Graph replay, compatibility, and correctness must be validated on the
exact host and device stack before enabling such a method.

Methods receive validated cache bindings. They must not mutate scheduler-owned
request objects or block tables. Returned lengths must be positive, no larger
than the declared maximum, and cover every layer when per-layer lengths are
used.

## Register another method

In-process registration:

```python
from vllm_ascend_kvcompress import register_method

register_method("my_method", create_method)
```

External packages may declare:

```toml
[project.entry-points."vllm_ascend_kvcompress.methods"]
my_method = "my_package.method:create_method"
```

Names are lowercase letters, digits, hyphens, or underscores. Duplicate names,
bad factories, and name mismatches fail closed.

## Acceptance checklist

- Parse only method-owned options and reject unknown values.
- Make all scheduler limits deterministic and block aligned.
- Validate model, RoPE, calibration, dtype, and cache layout before serving.
- Materialize every layer before returning from `compress`.
- Add registry, configuration, transaction, and numerical tests.
- Run matched compression-off/on correctness, quality, throughput, latency,
  and HBM acceptance on the exact supported host revisions.
