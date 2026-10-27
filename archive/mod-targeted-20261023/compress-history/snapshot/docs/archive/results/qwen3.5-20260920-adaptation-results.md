# Qwen3.5 historical adaptation experiments

English | [简体中文](qwen3.5-20260920-adaptation-results.zh.md)

> Historical 2026-09-20/27 engineering record; see the current design and benchmark protocol.

## Calibration and launch

Generate a model-bound artifact across two free devices:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m vllm_ascend_kvcompress.calibration \
  --model /workspace/models/Qwen--Qwen3.5-35B-A3B \
  --output artifacts/qwen3.5-35b-a3b-stats-v3.pt \
  --max-length 4096 --device npu:0 --device-map auto \
  --dtype bfloat16 --attn-implementation eager --local-files-only
```

The following B0/B1 recipe records the historical 2026-09-20 suite; it uses
`mamba_cache_mode=none` with APC and async scheduling disabled. It is not the
current 0.8 qualification launch. Configure the plugin with
[`examples/qwen3.5-35b-a3b-triattention.json`](../../../examples/qwen3.5-35b-a3b-triattention.json),
then launch the B1 service:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_PLUGINS=ascend,ascend_kvcompress
export VLLM_ASCEND_KVCOMPRESS_ENABLED=1
export VLLM_ASCEND_KVCOMPRESS_CONFIG=/absolute/path/qwen3.5-35b-a3b-triattention.json
# Required only for the validated host's legacy non-optional int[] GDN ABI.
export VLLM_ASCEND_KVCOMPRESS_QWEN_GDN_LIST_COMPAT=1
# Needed only when the container has /dev/shm=64 MiB; C4 uses a 40 MiB ring.
export VLLM_MQ_MAX_CHUNK_BYTES_MB=4
# Prepend the plugin source; do not replace the CANN Python path containing acl.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
vllm serve /workspace/models/Qwen--Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b --tensor-parallel-size 2 \
  --language-model-only --dtype bfloat16 --block-size 128 --mamba-cache-mode none \
  --max-model-len 32768 --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-async-scheduling \
  --generation-config vllm --enforce-eager
```

Although the command requests `--block-size 128`, the host reports the
32,768-token scheduler alignment and 2,048-token full-attention page separately.
The scheduler bind log records both, while the worker bind log also records the
128-token kernel block.

For B0, restart independently while retaining the same plugin and model-bound
statistics. Set `kv_budget` to 32,768 and `recompute_window` to 128, so the
32,896-token threshold is strictly above the service limit and no request is
compressed. B0 and B1 use the same command apart from this compression-budget
configuration. This keeps the compatibility hooks and runtime stack identical
without modifying either host repository. Never mix the two arms in one
service lifecycle.

The plugin applies an explicitly set `VLLM_MQ_MAX_CHUNK_BYTES_MB` to vLLM's
worker response queue as well as its scheduler broadcast queue. Messages above
the chunk size still use the socket path, so this reduces only the shared-memory
fast-path capacity and does not cap message size. Deployments without the
64 MiB constraint should omit the variable and retain the upstream default.

## Current 0.8 engineering status

On 2026-09-27, the synchronized hosts served Qwen3.5 BF16/TP=2/eager with a
16,384-token limit, `mamba_cache_mode=align`, APC, and async scheduling enabled.
Four non-thinking retrieval requests returned their expected codes, each with
one compression commit and one acknowledgement from each TP rank. The second
identical request added 8,192 server-side prefix-hit tokens. The two distinct
requests were submitted by concurrent clients, but overlapping device batches
were not established. See the [machine-readable smoke record](../../results/evidence/kvcompress-qwen35-20260927-async-apc-align-smoke.json)
and [frontier protocol](frontier-benchmarking-0.9.md). An early opt-in MTP2 trial
failed its first request in `aclnnCausalConv1d`. After correcting the target
cache-column mapping and extended rolling convolution state, one uncompressed
7K control and three compressed 10K retrieval requests all answered correctly,
with two TP acknowledgements per compression. MTP2 remains rejected by default
and is available only as an explicit experiment. Graph execution, 262,144-token
context, and the official 900-second workloads remain untested. See the
[diagnostic failure](../../results/evidence/kvcompress-qwen35-20260927-mtp2-diagnostic.json)
and [later engineering smoke](../../results/evidence/kvcompress-qwen35-20260927-mtp2-apc-align-smoke.json).
The CPU suite passes 135 tests with one environment skip.

## Historical 0.7 test status and scope

The 2026-09-20 worktree passed 114 unit/contract tests with one environment skip.
The 16K TP=2 service smoke completed in both arms: B0 retained all 16,384 input
tokens, while B1 committed 16,384 semantic tokens to 8,192 physical tokens;
both produced all 1,024 forced output tokens, matched the oracle, and reported
no silent truncation. The public-long-context and standard A2/A3 suites were
the historical model-level tests. Because this target used TP=2, eager execution,
the explicit legacy-ABI bridge, and hybrid attention, its results are
engineering evidence and cannot be labeled formal V4.6 acceptance. Results are
published in the
[HTML leaderboard](../../results/benchmark-leaderboard.html) with links to machine-readable
evidence and explicit limitations.

The public long-context run used a frozen `enable_thinking=false` chat template,
`--generation-config vllm`, and explicit neutral sampling controls. Each arm
completed 403 requests; all 806 requests succeeded without silent truncation:

| Dataset | B0 → B1 quality | Quality delta | Request throughput delta | Physical KV reduction | Commits / TP-rank acks |
| --- | ---: | ---: | ---: | ---: | ---: |
| LongBench-v2 (110) | 47.27 → 47.27 accuracy | 0.00 pp | −5.83% | 63.76% | 110 / 220 |
| passage_retrieval_en (200) | 100.00 → 100.00 | 0.00 pp | −2.39% | not applicable (short-output bypass) | 0 / 0 |
| Qasper (93) | 51.07 → 50.64 F1 | −0.43 pp | −1.28% | 42.11% | 11 / 22 |

All three quality-retention gates passed. See the
[machine-readable summary](../../results/evidence/kvcompress-working-tree-20260920-qwen35-public-summary.json)
for bootstrap intervals, latency changes, and raw hashes. Other NPUs on the
same host were executing the standard matrix, so throughput and latency deltas
are exploratory; quality, request integrity, and compression transactions are
the primary adaptation evidence.

## Transferred A2/A3 engineering matrix

The complete commissioning matrix was also transferred to this model. The
profile identifiers retain `FP16` to preserve workload identity, but Qwen3.5
actually ran in BF16, eager mode, TP=2, with the opt-in legacy GDN ABI bridge.
These runs are supplementary engineering evidence, not formal execution of the
standard graph-mode/TP=1 topology.

Each A2 cell used 16 measured requests per lifecycle, concurrency four, and
three independent cold lifecycles per arm:

| Shape | RPS | B0 → B1 median total tok/s | Change | Minimum physical KV reduction | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 8K + 512 | 0.05 | 249.72 → 250.80 | +0.43% | N/A: below threshold | PASS |
| 8K + 512 | 0.10 | 266.08 → 262.28 | −1.43% | N/A: below threshold | **FAIL** |
| 8K + 512 | 0.20 | 272.09 → 266.65 | −2.00% | N/A: below threshold | **FAIL** |
| 8K + 512 | 0.40 | 278.59 → 275.32 | −1.17% | N/A: below threshold | **FAIL** |
| 16K + 1,024 | 0.05 | 266.78 → 262.17 | −1.73% | 50.00% | **FAIL** |
| 16K + 1,024 | 0.10 | 274.73 → 269.07 | −2.06% | 50.00% | **FAIL** |
| 16K + 1,024 | 0.20 | 278.00 → 274.01 | −1.43% | 50.00% | **FAIL** |
| 16K + 1,024 | 0.40 | 275.39 → 273.54 | −0.67% | 50.00% | PASS |

All 768 measured arm-requests and six warm-ups completed correctly, with zero
failures, short forced outputs, or silent truncations. The 16K B1 cells recorded
192 scheduler commits and exactly 384 per-rank acknowledgements. Two cells
passed; six failed only the frozen 1% total-throughput regression budget.

A3 used 30,720 input tokens plus 2,048 forced output tokens, one request at a
time, a five-minute warm-up, and a 30-minute measurement split into six windows:

| Median of three lifecycles | B0 | B1 | Change |
| --- | ---: | ---: | ---: |
| Total token throughput | 69.34 tok/s | 70.19 tok/s | +1.23% |
| Mean TTFT | 1949.71 ms | 2009.13 ms | +3.05% |
| Mean TPOT | 229.91 ms | 227.05 ms | −1.24% |
| Mean E2E | 472.57 s | 466.77 s | −1.23% |

Every B0 and B1 lifecycle completed four correct requests, below the required
24. Every six-window throughput CV was 100%, above the 5% gate, and empty
windows made latency-drift statistics unavailable. B1 nevertheless preserved
73.33% minimum physical-KV reduction with 24 commits and 48 TP-rank
acknowledgements. A3 is therefore **FAIL**, despite passing quality,
transaction, reduction, and throughput-regression checks.

Three initial sandbox launches could not enumerate Ascend devices and failed
before model loading or requests. Their log hashes are retained but excluded
from the three valid lifecycles. Full hashes and per-cell values are in the
[machine-readable standard-matrix summary](../../results/evidence/kvcompress-working-tree-20260920-qwen35-standard-summary.json).

This does not negate the V3 paper's hybrid-model warning: its Qwen3.5-27B and
35B-A3B V3-only runs failed strict middle/end NIAH, while the later long-context
rescue was not validated on those larger models. This run's
`passage_retrieval_en` workload requests only 32 output tokens and therefore
bypasses compression by design; it proves the bypass, not post-compression
needle retention. Deployment still needs an authorized multi-position NIAH or
production retrieval gate under active compression. The present conclusion is
strictly limited to the public tasks reported above.
