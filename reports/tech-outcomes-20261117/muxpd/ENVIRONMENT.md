# Historical environment and replay provenance

This report recovers the wl38 performance measurements from 2026-11-07/08. The archive operation
made no new model requests or performance measurements. The scope is Qwen3.5-35B-A3B BF16, TP2/TP4,
MuxPD default mode versus native, one pass per configuration and workload. It does not measure
answer quality or ROW mode.

[provenance.json](provenance.json) records structured identities, source hashes and a timestamped
current hardware inventory. [originals-manifest.json](originals-manifest.json) binds preserved files
to their original paths and bytes; gzip results decode to the recorded original bytes.

## Stack and actual configuration

All four [run identity records](data/original/runs/) record the same Git identities:

| Component               | Recorded identity                                                             |
| ----------------------- | ----------------------------------------------------------------------------- |
| MuxPD                   | `43fcaf4279640bcc086119b0671cb7fd4228836a`; `dirty=0` for `src` and `scripts` |
| vLLM core               | `d0f22d2bda562156e4dbf433ce645e1769b4f804`                                    |
| vLLM-Ascend             | `c182a9ed1ed8ca773512d6549e15de6fd471aa00`                                    |
| Model                   | Local `/root/workspace/models/Qwen3.5-35B-A3B`; BF16; language model only     |
| Server-reported version | `0.1.dev1+gd0f22d2bd`                                                         |
| torch-npu               | Server compiler log: `2.10.0.post4`                                           |

The identity records also contain `VLLM_VERSION=0.25.1`. This is an environment value, not the
version reported by the running server: both server logs explicitly warn that `VLLM_VERSION` is
unknown. Use the Git SHA and server banner above for runtime attribution. Identity capture records
core/Ascend HEADs but does not record their dirty status or prove exact installed-package bytes.

The four preserved server logs show the effective startup arguments, including the resolution of
duplicate flags in the composed launch scripts:

- [TP2 MuxPD](data/original/runs/wl38-mux-tp2/server.log) and
  [TP2 native](data/original/runs/wl38-nat-tp2/server.log).
- [TP4 MuxPD](data/original/runs/wl38-mux-tp4/server.log) and
  [TP4 native](data/original/runs/wl38-nat-tp4/server.log).

| Setting       | Both arms                                                                                                                  | Arm-specific behavior                                                                                             |
| ------------- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| Parallelism   | TP2 on visible devices `0,1`; TP4 on `0,1,2,3`; expert parallel enabled                                                    | One service at a time, port `8300`                                                                                |
| Capacity      | `max-model-len=16384`, `max-num-seqs=16`, `max-num-batched-tokens=16384`                                                   | MuxPD accepts a full prefill; native uses chunked prefill                                                         |
| Execution     | Async scheduling; seed `0`; BF16; prefix caching disabled; memory utilization `0.85`                                       | MuxPD uses `AsyncDecoupledMuxPDScheduler` and `AsyncDecoupledMuxPDNPUWorker`                                      |
| Graphs        | Compile mode `3`, `FULL_DECODE_ONLY`, capture sizes `[1,2,4,8,16]`, maximum `16`                                           | MuxPD uses decoupled execution, isolated overlap communication, four layers per prefill slice, and a decode graph |
| Host settings | `OMP_NUM_THREADS=8`, worker method `spawn`; recorded plugins `ascend,ascend_kv_connector,ascend_model,ascend_model_loader` | MuxPD sets `VLLM_USE_V2_MODEL_RUNNER=0`, disables NUMA binding and CPU binding in its script                      |

[serve-qwen35-muxwise.sh](data/original/plugin-scripts/serve-qwen35-muxwise.sh) composes
[serve-qwen35.sh](data/original/plugin-scripts/serve-qwen35.sh); both preserved files match the
named MuxPD commit byte for byte. The native launcher is
[serve-native.sh](data/original/tools/serve-native.sh). Values such as the base scripts' sequence
limit `8` and graph maximum `8` are overridden by the final run arguments; the effective values
above are corroborated by the logs.

The [driver log](data/original/wl38/driver.log) records keepalive stopped throughout the experiment
and restored at the end. The [run script](data/original/wl38/run.sh) unsets `HCCL_DETERMINISTIC` and
`VLLM_BATCH_INVARIANT`; neither is present in the recorded allowlisted identity environments.

## Hardware and model identity boundaries

The earlier
[fixed wl37 environment report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/2c364bbfe228984ae63fd23210cfd04c3b1fad00/reports/muxpd-perf-workload-default-vs-native-20261107/REPORT-v1.md)
identifies the same stack/environment as Ascend 910B2. Historical wl38 identities preserve device
numbers and TP sizes; the raw results preserve NPU worker-memory lines. They do not contain a
historical device-model, driver or firmware inventory.

During this archival audit, a read-only `npu-smi info` query confirmed the current host has eight
910B2 NPUs, each with 65536 MB HBM, and reports npu-smi/version `26.0.rc1`. Its exact UTC capture
time and stdout are embedded in [provenance.json](provenance.json). This is current inventory, not
evidence of the driver or device state during the historical run.

Historical engine logs show `revision=None` and `tokenizer_revision=None` for the local model. The
current model directory's 27 Hugging Face download metadata files consistently name commit
`59d61f3ce65a6d9863b86d2e96597125219dc754`, including all 14 weight shards and
tokenizer/template/config files. These metadata records support current-cache provenance. They are
not run-time byte hashes and cannot establish that those exact bytes were loaded on November 7/8.

Missing historical identity material is recorded explicitly in `provenance.json`: model-shard
digests, tokenizer/config/template digests, the complete `runtime/env.sh` and launch environment,
CANN/driver/firmware/OS inventory, and an installed Python dependency/wheel lock. The archived
configuration is sufficient to inspect the reported settings and recompute metrics from recorded
requests; a bit-identical full device replay is not established.

## Time and order

Full UTC timestamps in the driver log anchor the window. Dates for its time-only completion lines
follow the midnight rollover; raw request timestamps are monotonic `perf_counter()` values, not
wall-clock dates. Beijing is UTC+08:00.

| Event                                 | UTC                 | Beijing             |
| ------------------------------------- | ------------------- | ------------------- |
| Keepalive stopped / experiment begins | 2026-11-07 23:50:58 | 2026-11-08 07:50:58 |
| TP2 MuxPD `wl-2wiki-r1.6` completed   | 2026-11-07 23:59:09 | 2026-11-08 07:59:09 |
| TP2 native `wl-2wiki-r1.6` completed  | 2026-11-08 00:16:54 | 2026-11-08 08:16:54 |
| TP4 MuxPD `wl-2wiki-r1.6` completed   | 2026-11-08 00:33:53 | 2026-11-08 08:33:53 |
| TP4 native `wl-2wiki-r1.6` completed  | 2026-11-08 00:51:00 | 2026-11-08 08:51:00 |
| Keepalive restored / experiment ends  | 2026-11-08 00:59:47 | 2026-11-08 08:59:47 |

Order was TP2 MuxPD → TP2 native → TP4 MuxPD → TP4 native. Each arm replayed six plans once, in the
order preserved by the run script. The two configurations were not interleaved or randomized; this
single-pass record does not separate temporal drift from configuration effects.

## Fixed data and workload construction

The [pre-run plan](data/original/wl38/PLAN.md), [builder](data/original/wl38/tools/build_plans.py),
[build log](data/original/wl38/build.log),
[source/selection manifest](data/original/wl38/plans/MANIFEST.json) and
[plan hashes](data/original/wl38/plans/SHA256SUMS) preserve the workload specification. The full
source revisions come from the earlier fixed wl37 report and the benchmark repository's fixed
[LongBench source documentation](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/a3d96977005f7257502ded2938a051c70de60b02/docs/workload-data/LONGBENCH_V1_QASPER_ORIGINAL.zh-CN.md)
and
[OASST source documentation](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/a3d96977005f7257502ded2938a051c70de60b02/docs/workload-data/OASST1_ORIGINAL_CONTINUATION.zh-CN.md).
The wl38 builder checks exact file hashes before constructing plans.

| Source artifact             | Fixed source                                                                                                                                                                                                                                   | SHA-256 expected by builder                                        |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `longbench-data.zip`        | [THUDM/LongBench data.zip at `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`](https://huggingface.co/datasets/THUDM/LongBench/resolve/5e628be450b7e67fb7ae6e201bd6d8f7056f7672/data.zip)                                                            | `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64` |
| `dataset2prompt.json`       | [LongBench source `2e00731f8d0bff23dc4325161044d0ed8af94c1e`](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2prompt.json)                                                           | `56d22ad4f382169c2b8a11ff4c982a4a1bea096c8152b0f0b85b64686b157c30` |
| `dataset2maxlen.json`       | [Same LongBench source commit](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2maxlen.json)                                                                                          | `72966b3c0933e214591637fb085798c5e687ebff4ddaab5d99bbc31120532022` |
| `oasst1-validation.parquet` | [OpenAssistant/oasst1 at `fdf72ae0827c1cda404aff25b6603abec9e3399b`](https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/fdf72ae0827c1cda404aff25b6603abec9e3399b/data/validation-00000-of-00001-134b8fd0c89408b6.parquet), 4401 rows | `24002597bb13a7edd42d92f773762f25e285f72c31a70449393d0ded1dc7b416` |
| `oasst1-validation.jsonl`   | Prior conversion: pyarrow `18.1.0`, `to_pylist()`, one `json.dumps(row, default=str)` line per row                                                                                                                                             | `3f1290a7f2b821516f7f7f66d7dcbf87816779656a35a6def084d26548fb2cec` |

All five current local source files were rehashed during the archive audit and match these builder
expectations. The archive does not redistribute source bodies or reference answers.
`provenance.json` additionally identifies the exact `data/2wikimqa.jsonl` ZIP member.

`wl38/wl-2wiki-r1.6` uses nominal Poisson arrivals at 1.6 requests/s for 120 s, seed `38`, with
component probabilities 20% LongBench `2wikimqa` and 80% OASST1 validation. The fixed realization
has 229 requests: 45 long and 184 short. Selected long inputs have median 6257 tokens. The five
LongBench rows exceeding `input + official budget <= 16384` are excluded and listed in the original
manifest; inputs are not truncated.

LongBench official prompts are wrapped as a single user message. OASST inputs contain the complete
valid alternating ancestor chain before the selected assistant target. Both use the served
tokenizer/chat template with `enable_thinking=false`. Output budgets are reference token counts
clipped to `[16, official budget]` for LongBench or `[16,2048]` for OASST, and are forced by
`ignore_eos=true`. Each source pool is shuffled once using its pool name and seed `38`; all six
plans consume consecutive disjoint slices, so rebuilding only this workload independently would
change selection. Arrival times, composition and output lengths are test settings. This is a
real-text-derived performance workload, not production arrivals, official task execution or
answer-quality evidence.

The [derived request manifest](data/request-manifests/wl-2wiki-r1.6.json) preserves arrival, source
row/ID, kind, input/output lengths and a hash of each prompt token array, while omitting the arrays
themselves. The original full-plan hash is
`4054d9fc03a41435029e51baf69fc4c9bc26635a5e586c6b038b9deef33076d0`. Raw results are
[TP2 native](data/results/wl38-tp2/native/pass1/wl-2wiki-r1.6.json.gz) and
[TP2 MuxPD](data/results/wl38-tp2/mux43/pass1/wl-2wiki-r1.6.json.gz); `wl-docchat-r1.6` is a
separate wl37 workload and its archived numbers must remain separate.

## Commands and metric replay

The exact historical command sequence is preserved in [run.sh](data/original/wl38/run.sh), with
[launch.sh](data/original/tools/launch.sh), the two MuxPD scripts and the native script linked
above. The effective TP2 launches were the following calls within that run script; `arm` is its
recorded helper and `G16` is its graph JSON:

```sh
MUX=/root/workspace/vllm-ascend-MuxPD-hust/scripts/serve-qwen35-muxwise.sh
NAT=/root/workspace/muxpd-eval/tools/serve-native.sh
G16='{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16],"max_cudagraph_capture_size":16}'
arm wl38-mux-tp2 mux43 0,1 2 wl38-tp2 "$MUX" --served-model-name qwen3.5-35b --seed 0 --max-num-seqs 16 --compilation-config "$G16"
arm wl38-nat-tp2 native 0,1 2 wl38-tp2 "$NAT" --max-num-seqs 16 --max-num-batched-tokens 16384 --compilation-config "$G16"
```

The load-generation call for each selected arm was:

```sh
timeout 2400 /root/workspace/.venv/bin/python tools/loadgen.py run \
  --plan wl38/plans/wl-2wiki-r1.6.json --port 8300 \
  --out "results/wl38-tp2/$label/pass1/wl-2wiki-r1.6.json" --label "$label"
```

Here `$label` was `mux43` or `native`. The load generator makes two warmup requests before starting
`t0`; it uses temperature `0`, streaming, usage reporting, `ignore_eos=true`, a 1800 s per-request
timeout, and records failures without retry. The service-health timeout is 1800 s. The preserved
scripts contain historical absolute paths, environment sourcing and service/keepalive operations.
They document the historical commands and were not executed by this audit. A full replay requires
the original authorized source inputs and a separately prepared compatible runtime/model
environment, with the identity gaps above retained.

Metric recomputation uses only the archived request receipts, with the definitions in
[loadgen.py](data/original/tools/loadgen.py) and [aggregate.py](data/original/tools/aggregate.py):
TTFT runs from scheduled arrival to first nonempty SSE event, including send lag; TPOT is
`(last event - first event)/(usage completion tokens - 1)`; gap percentiles pool consecutive
nonempty SSE receipt intervals. SLO passing requires all of TTFT ≤1 s for short or ≤3 s for long,
TPOT ≤100 ms, and maximum receipt gap ≤500 ms, with no request error.

The historical field `goodput_frac` means SLO-passing requests divided by all requests.
Successful-request rate is separate. Goodput requests/s and output token throughput use
`last successful event - t0`, including queue drain, not the nominal 120 s arrival window or the
result's slightly later `end`. Both TP2 arms produced 29880 output tokens; spans are 184.812266912 s
native and 149.873492600 s MuxPD.

SSE events are client receipts with nonempty text, not guaranteed one event per generated token.
Event count differs from usage completion-token count on 55 native and 57 MuxPD requests in the
selected TP2 pair. Describe these intervals as SSE/event gaps; the data cannot reconstruct exact
internal token emission times. Generated response text was not retained, so this archive cannot add
a quality assessment.
