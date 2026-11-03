# Qwen2.5-14B matched PyramidKV retest — 2026-11-03

This is a real **Eager diagnostic reproduction**, outside official leaderboard specs and snapshots.
It answers whether the current independent plugin can recover the historical Qwen2.5-14B,
7168-input/512-output, concurrency32 effect.

All six fresh server lifecycles succeeded: OFF/ON repeated three times on physical Ascend910B2 card6
(container ordinal1). The480 measured requests and12 separately labelled readiness requests are
preserved. Metrics below are arithmetic means of three per-process metrics, matching the historical
aggregate convention.

| Long workload metric           |      OFF |       ON |            ON/OFF change |
| ------------------------------ | -------: | -------: | -----------------------: |
| Output throughput, tokens/s    |  90.0999 | 156.8658 |                  +74.10% |
| Request throughput, requests/s | 0.175976 | 0.306379 |                  +74.10% |
| Mean TTFT, seconds             |  44.7534 |  17.3745 |                  -61.18% |
| Mean TPOT, ms                  | 118.3613 | 156.0968 |                  +31.88% |
| Mean E2E, seconds              | 105.2360 |  97.1400 |                   -7.69% |
| Sampled KV usage peak          |  100.00% |   43.10% | -56.90 percentage points |
| Preemptions per long cohort    |        2 |        0 |                       -2 |
| Card HBM peak, MiB             |  53714.0 |  53841.3 |                +127.3MiB |

The automatic KV pool is892 blocks /114176 tokens /20.90625GiB in both arms. The plugin frees active
KV ownership inside that preallocated pool; it does not return the pool allocation to the device.
Actual HBM therefore rises slightly. All33 long-prompt compression transactions per enabled
lifecycle (one readiness plus32 measured) have matching scheduler commits and worker
acknowledgements; 7168 semantic tokens become1024 physical tokens,56 source pages become8 private
destination pages. Every lifecycle ends with zero active/waiting requests and an idle card after
shutdown. Per-run metrics, including short cohorts and percentiles, are in
[analysis.json](analysis.json).

## Identity and comparison limits

- Candidate source:
  [c24a0eb82df54731720a5b4bc81c23ed047c0f3f](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/commit/c24a0eb82df54731720a5b4bc81c23ed047c0f3f).
  The public0.4.0rc1 wheel rejected Qwen2.5. This source candidate adds the exact dense eager
  profile and fixes its five-argument cache-allocation host ABI. The archive contains the measured
  wheel, source diff and hashes. Its package metadata still says0.4.0rc1; it is **not** the
  previously published wheel.
- Core`d0f22d2bda562156e4dbf433ce645e1769b4f804`, Ascend`03766ac696fde5ab1980d80ca0b8543d3580c989`;
  both clean.
- User-approved current CANN9.1 / Torch2.10 / torch-npu2.10.post4 in both arms. Historical CANN9.0 /
  Torch2.9 was not restored. Driver26.0.rc1; one910B2.
- Official Qwen2.5-14B-Instruct, BF16, TP1/PP1, context8192, seq32, batched tokens2048,
  utilization0.8, block128, seed0, APC/async/chunked, enforce-eager, generation-config=vllm, no
  speculation. All model shards and tokenizer/config/index are hashed in`model-identity.json` inside
  the archive.
- PyramidKV512/beta20/window8/kernel7/maxpool, kv_head/mean, no merge, admission strictly above4096
  tokens. Both arms use the same candidate package, explicit host worker and launch; only activation
  differs.
- Each lifecycle runs48×1024/256 at rate1, then32×7168/512 at rateinf/C32. The pinned legacy
  core4861aab3 generator is executed and its seed0 request texts are frozen. All actual server
  input/output counts match the requested counts. Each cohort has one first-prompt readiness request
  and no extra warmup.
- SSE timestamps preserve legacy first-choice TTFT / last-choice E2E semantics, including empty-text
  token choices; TPOT=(E2E−TTFT)/(output_tokens−1). Metrics sampled approximately every0.2s; HBM
  approximately every2s. Completed-request serialization/writes run in the client event loop and are
  included in cohort wall time; this small instrumentation overhead is common to both arms. The
  historical client did not preserve those per-request files.

The historical +41.38% throughput / -30.98% TTFT /99.93%→42.42% KV result is recovered
directionally, with larger throughput/TTFT improvements in this current runtime. This is not a claim
that either runtime is globally faster. The same tradeoff remains: higher TPOT. This synthetic
performance test does not establish current task-quality equivalence, general support for other
models, or a maximum admissible concurrency. ON preflight had no live preemption; OFF formal runs
did.

## Audit and reproduction

`raw.tar.gz` holds immutable requests, raw SSE and timestamps, sampled full metrics/HBM text,
server/client logs, launch scripts, model/runtime identities, preflight failures/correctness
receipts, the frozen contract, and offline analysis code. `members.json.gz` binds every member by
SHA256 and byte size. The initial unsupported dense cache-allocation preflight is retained. The code
suite passed258 tests with3 skips; the real NPU40/8 GQA oracle passed. A stalled SSE test verifies
cancellation preserves partial responses. Failed/partial formal runs would have stopped the series;
none occurred.

Run the CPU-only public verifier with Python3.12:

```bash
python verify.py
```

It checks outer/member hashes, extracts into a temporary directory, replays all 480 measured
streams, verifies the exact payloads/usage, recomputes every per-request and cohort metric, checks
transactions/cleanup, and compares the result byte-for-byte with the published analysis. No model
weights, NPU, PyTorch, network access or private inputs are required for this verification.
Re-executing device runs requires the pinned prepared Ascend runtime and model files. The frozen
absolute-path controller/launch files document this host; adapt paths before freezing a new
experiment, never overwrite these originals.
