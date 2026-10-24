# Draft B16 performance recovery handoff (2026-09-14)

## Goal and comparison protocol

Recover serial Draft performance to at least the current EAGLE result under the
engineer-provided online protocol:

- Dataset: GSM8K materialized custom dataset, 200 prompts
- Target: Qwen2.5-14B-Instruct, Draft: Qwen2.5-0.5B-Instruct
- Batch / max-num-seqs: 16
- Maximum output length: 256
- Dtype: FP16
- max-model-len: 32768
- max-num-batched-tokens: 8192
- Prefix caching: disabled
- Graph mode: FULL_DECODE_ONLY
- Generation config: auto (includes repetition_penalty=1.05)

Reference results from the same online protocol:

| Method | Output throughput | Duration | Speedup vs baseline |
| --- | ---: | ---: | ---: |
| Baseline | 454.10 tok/s | 93.1919 s | 1.000x |
| EAGLE Adaptive | 611.27 tok/s | 69.1432 s | 1.346x |
| Draft Adaptive, gamma 1..4 | 172.35 tok/s | 245.6427 s | 0.380x |
| Draft fixed gamma 5 | 228.69 tok/s | 184.86 s | 0.504x |

The target for resuming this work is Draft >= 611.27 tok/s under a fair B16
protocol.

## Main finding

The low B16 Draft result is not an inherent small-batch limitation. The old
vLLM-HUST stack reproduced the serial Draft path at **724.2821 output tok/s**:

- 200 GSM8K prompts, B16, gamma=5
- 60,366 output tokens in 83.34598 s
- Strict Draft acceptance behavior; historical AC was about 79.76%
- Old stack/config: BF16, max output 512, max-model-len 1024,
  max-num-batched-tokens 32848, prefix cache enabled,
  FULL_AND_PIECEWISE graph mode
- The startup log explicitly reports `Asynchronous scheduling is enabled.`

This proves a real current-stack/configuration regression. It also identifies
the highest-priority untested difference: the `arc-easy` plugin protocol sets
`async_scheduling = false`, while the successful old run used asynchronous
scheduling. The engineer's raw baseline command did not explicitly disable
async scheduling.

Old-stack reproduction command:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH=/root/data/vllm-hust:/root/data/vllm-ascend-hust:$PYTHONPATH
export ASCEND_RT_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DATASET_MODE=gsm8k NUM_SAMPLES=200 MAX_TOKENS=512 MAX_MODEL_LEN=1024
export MAX_NUM_SEQS=16 NUM_SPECULATIVE_TOKENS=5 EXECUTION_MODE=graph
export BENCHMARK_KIND=draft RUN_BASELINE=0
/root/miniconda3/envs/vllm-hust-dev/bin/python \
  /root/data/vllm-ascend-hust/test_qwen25_draft_offline.py
```

## Current-stack results

Removing the accidentally inherited `TORCHDYNAMO_DISABLE=1` improved current
FP16 fixed-gamma=4 performance, but did not recover it:

- 50-prompt screen: 258.94 tok/s
- Full 200 prompts: 247.7765 tok/s, 42,322 output tokens, 170.8072 s

Best non-comparable 50-prompt screens reached about 313 tok/s, including a
confidence-margin experiment. They remain far below the EAGLE gate and should
not be reported as the recovered result.

Result directory:

```text
/root/data/vllm-hust-vSpec/benchmark_results/b16_draft_recovery_20260913
```

Important files:

```text
compiled_g4_fp16_n200.json
screen/compiled_g4_fp16.json
screen/static_g4_fp16.json
screen/torchair_g4_fp16.json
screen/super_g4_fp16.json
screen/bf16_g5.json
screen/full_g5.json
```

## Paths already ruled out

These paths did not recover throughput and should not be repeated unchanged:

| Experiment | 50-prompt output throughput | Outcome |
| --- | ---: | --- |
| NPUGraphEx static-kernel attempt | 195.00 tok/s | Worse; CANN child compiler also lacked numpy |
| TorchAir (`enable_npugraph_ex=false`) | 213.09 tok/s | Worse |
| ACL graph super-kernel | 219.35 tok/s | Worse; temporary patch removed |
| Event-ordered replay | 222.51 / 236.98 tok/s | Worse |
| Draft-body W8A16 | about 236-280 tok/s | No recovery |
| Active-vocabulary screens | about 222-287 tok/s | No recovery |
| gamma=8 / gamma=12 | 303.04 / 252.14 tok/s | No recovery |

Static-kernel output was written under
`static_kernel_compile_outputs/`; no static-kernel code was retained.

## Working-tree state

No benchmark processes or vLLM servers were left running when this handoff was
written. The following implementation work is intentionally uncommitted:

```text
 M src/vllm_hust_vspec/backends/draft.py
 M src/vllm_hust_vspec/backends/draft_vocab.py
 M src/vllm_hust_vspec/backends/eagle_body_quant.py
 M src/vllm_hust_vspec/backends/eagle_rejection.py
 M src/vllm_hust_vspec/cli.py
 M tests/test_launcher.py
?? src/vllm_hust_vspec/backends/draft_body_quant.py
```

Relevant retained changes include:

- Reuse the existing Draft `ACLGraphWrapper` and mark merged replay with
  `use_eagle=True`, avoiding its replay barrier.
- Correct active-vocabulary monkey patches for Draft logits/gather paths.
- Penalty-aware linear rejection optimization.
- Experimental Draft-body W8A16 support, disabled by default because it did
  not improve this workload.

Targeted unit tests previously passed: 15 tests.

## Resume order

1. Run current-stack fixed gamma=5 with `--async-scheduling` and no Adaptive.
   Keep all engineer protocol settings otherwise unchanged. This is the most
   likely immediate recovery based on the successful old run.
2. If positive, run baseline and EAGLE with the same async setting so the
   comparison remains fair, then enable `--adaptive-speculation` together with
   `--adaptive-async` and repair any state-ordering failure.
3. Test current Draft with `FULL_AND_PIECEWISE`. The old 724 tok/s run used this
   mode, while the plugin CLI currently exposes only `piecewise`,
   `full-decode-only`, and `full`.
4. Complete the interrupted old-stack same-shape diagnostic: output 256,
   max-model-len 32768, prefix cache off. This separates model-length/prefix
   effects from scheduler/version effects.
5. If async and graph mode do not close the gap, profile old and current stacks
   side by side around Draft graph replay, graph task updates, and
   `_copy_draft_token_ids_to_cpu`. Previous current-stack profiling showed
   Draft replay and graph metadata updates dominate the iteration.

Current-stack environment must include all three source roots:

```bash
source /opt/vllm-hust-cann91/Ascend/cann-9.1.0/set_env.sh
export PYTHONPATH=/root/data/vllm-hust-vSpec-github/src:/root/data/vllm-hust-latest:/root/data/vllm-ascend-hust-latest:$PYTHONPATH
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Do not set `TORCHDYNAMO_DISABLE=1` for the recovery runs.

## Checkpoint update (2026-09-15 04:15 UTC)

This section supersedes the earlier resume order. The serial Draft path has
been recovered substantially, but it has not yet reached the EAGLE gate under
the engineer-provided online B16 protocol.

### Comparable results

All rows below use FP16, GSM8K, output length 256, B16,
`FULL_DECODE_ONLY`, async scheduling, generation config `auto`, prefix cache
off, and the engineer's materialized dataset.

| Method / experiment | Prompts | Output tokens | Duration | Output throughput |
| --- | ---: | ---: | ---: | ---: |
| Target-only reference | 200 | 42,318 | 93.1919 s | 454.10 tok/s |
| EAGLE reference | 200 | 42,265 | 69.1432 s | 611.27 tok/s |
| Draft exact, fixed gamma 2, best full run | 200 | 42,282 | 77.3557 s | 546.5919 tok/s |
| Draft exact, fixed gamma 2, best N50 screen | 50 | 10,610 | 18.5150 s | 573.0474 tok/s |
| Draft exact clean rerun, profiled | 50 | 10,610 | 19.8813 s | 533.6682 tok/s |
| Draft exact clean rerun, same server repeat | 50 | 10,614 | 19.1123 s | 555.3483 tok/s |
| Draft exact, historical capture buckets | 50 | 10,607 | 19.3772 s | 547.3956 tok/s |
| Draft exact, Target FIA update workers 3 | 50 | 10,610 | 18.6683 s | 568.3443 tok/s |

The 573 tok/s N50 value is a valid fixed-gamma result, but it is a high screen
result rather than the full-run gate. The current reportable Draft result is
the N200 value, 546.5919 tok/s: 1.204x Target-only and 0.894x EAGLE.

The explicit historical capture list
`1 2 3 4 5 6 8 10 12 16 20 24 32 40 48 64 80` did not improve throughput,
so the remaining gap is not caused primarily by the newer automatic graph
bucket list.

### Profile and current bottleneck

The clean gamma-2 profile is stored at:

```text
/tmp/vspec_draft_clean_g2_p4_profile.json
```

Important mean host-call times across two N50 passes:

| Phase | Mean time |
| --- | ---: |
| Draft proposal | 22.196 ms |
| Draft FIA graph update | 8.211 ms |
| Target FIA graph update | 12.161 ms |
| Target sampling | 4.110 ms |
| Target graph replay call | 2.520 ms |

The Draft and Target FIA graph-task rebinding remains the largest structural
cost. Target sampling is the next actionable cost because the model's
`generation_config.json` applies `repetition_penalty=1.05` over the full
152,064-token vocabulary.

A diagnostic run with `--generation-config vllm` reached 584.4309 tok/s in
17.5675 s, but produced only 10,267 output tokens and is therefore not a fair
result. It is retained only as evidence that an exact, cheaper repetition
penalty path can recover several milliseconds per decode step.

### Rejected repetition implementation

`VSPEC_DRAFT_FUSED_REPETITION=1` currently enables an experimental persistent
full-vocabulary seen-mask implementation in `draft_repetition.py`. It reached
only 283.6537 tok/s. Its full `[batch, vocab]` row gather and FP32 logits copy
spill asynchronous work into the following Draft proposal. Leave this flag
unset; this implementation must not become a default or release feature.

The earlier sparse top-8 implementation is also rejected as currently
written. It rebuilds padded token histories every step and reached only
539.0618 tok/s.

### Next implementation target

1. Implement exact repetition-aware greedy selection over a bounded raw-logit
   top-K candidate set, backed by persistent per-request seen-token state.
   Select the best unseen raw candidate and the best penalty-adjusted seen
   candidate. If all K candidates are seen, use the K-th raw score as an upper
   bound and fall back to the generic processor only when that bound cannot
   prove the selected seen candidate is globally optimal.
2. Keep the candidate path opt-in until token IDs match the generic processor
   on randomized and real GSM8K traces. Measure K=64/128/256 before choosing a
   default.
3. Combine the successful Target FIA setting
   `VSPEC_DRAFT_TARGET_PARALLEL_GRAPH_UPDATES=3` with cached normalized graph
   descriptors to remove per-step sorting, tuple normalization, list slicing,
   event creation, and executor submission overhead.
4. Use N20/N50 only for rejection. Run the full N200 protocol only after an
   N50 result exceeds 611.27 tok/s with identical output-token semantics.

### Exact fixed-gamma launch configuration

The last clean runs used these important controls:

```bash
export VSPEC_DRAFT_PARALLEL_GRAPH_UPDATES=4
unset TORCHDYNAMO_DISABLE
unset VSPEC_DRAFT_PROFILE_PATH
unset VSPEC_DRAFT_FUSED_REPETITION
unset VSPEC_DRAFT_ALIGN_REPETITION
unset VSPEC_DRAFT_TARGET_TENSOR_FIA
unset VSPEC_DRAFT_TENSOR_FIA

vllm-hust-vspec \
  --method draft \
  --target-model /data/shared-models/Qwen2.5-14B-Instruct \
  --draft-model /data/shared-models/Qwen2.5-0.5B-Instruct \
  --served-model-name qwen2.5-14b-draft-vspec-fp16 \
  --gamma 2 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --max-model-len 32768 \
  --dtype float16 \
  --block-size 128 \
  --gpu-memory-utilization 0.85 \
  --host 127.0.0.1 \
  --port 18183 \
  --graph-mode full-decode-only \
  --generation-config auto \
  --async-scheduling \
  --no-prefix-caching \
  --chunked-prefill \
  --no-adaptive-speculation \
  --draft-lm-head-quantization w8a16 \
  --graph-event-ordering \
  -- \
  --additional-config '{"enable_reduce_sample":true}'
```

Results from this checkpoint are under:

```text
benchmark_results/draft_b16_recovery_20260914/
```

No vLLM server or benchmark process was left running at this checkpoint. The
working tree remains intentionally uncommitted so the retained optimization
patches and rejected opt-in candidates can be separated during the next pass.

Evidence-path note: historical `benchmark_results/` inputs are preserved in the
[fixed benchmark originals](benchmark-originals.md). Download the verified external
snapshot before reading these paths; missing local datasets are not reconstructed by migration.
