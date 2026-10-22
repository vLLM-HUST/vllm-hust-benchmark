# Qwen3.5 similarity workload: latency-priority policy

Measured 2026-10-08 with `/data/jxd/Qwen3.5-35B-A3B-W8A8` on Ascend 910B3,
TP1. Baseline used card 7, port 8107; SimLLM used card 6, port 8106.
Both used context length 8192, block size 128, GPU memory utilization 0.85,
max sequences 4, PIECEWISE capture size 4, and disabled vLLM prefix caching
and chunked prefill. Requests used chat completions, temperature 0, Qwen
thinking disabled, one output token, and global client concurrency 16.

The input is `resources/workload/final_workloaddata_v2/semantic_similar.jsonl`.
The fixed 300-request sample contains 60 complete five-request groups from
that file. The plugin run followed a 30-request warmup sample, warming its
task cache. The baseline does not use SimLLM. Each run below is one observation
on a separate card; small differences need replication.

| Policy | Throughput (requests/s) | TTFT p50 (s) | TTFT p95 (s) | Reuse hits / 300 |
| --- | ---: | ---: | ---: | ---: |
| Fresh baseline | 5.6487 | 2.7893 | 3.1036 | 0 |
| `tail`, context topics exact, suffix 512 | 5.8923 | 2.6827 | 3.1818 | 112 |
| `tail_anchor`, main topic and template, suffix 512 | 7.0352 | 2.2373 | 2.5120 | 297 |

`tail_anchor` exceeded the requested 10% small-sample target: throughput rose
24.55%, median TTFT fell 19.79%, and p95 TTFT fell 19.06% relative to the
fresh baseline. It logged 698,949 reused tokens in 297 requests, with no
SimLLM KV extraction or injection failures. This build's wheel SHA-256 is
`4163fd70236fc94c88d94ba47c4ddabadb73cddf5cdbbae4b533c4a4f8112092`.
The measured service explicitly set `HYBRID_REUSE_MODE=tail_anchor` and
`HYBRID_RECOMPUTE_TOKENS=512`. After the full run, the same values became
the configuration defaults in a new wheel, SHA-256
`f27eefb4638c394791e211191a35d1fae80af76eea8ca548860dc28f34d99c46`.
The default-setting change was checked in the installed remote package; it
does not change the measured inference path.

Local non-model tests: 191 passed, 2 skipped, 4 deselected. Remote non-model
tests: 191 passed, 2 skipped, 4 deselected; the manifest test was excluded
because this remote environment lacks `vllm_hust_ext`. Ruff passed on `src`
and `tests`. The remote suite was repeated after installing the final
default-setting wheel with the same result.

## Complete 5,000-request replay

The full similarity workload was replayed sequentially using the same chat
request settings. Both output files contain 5,000 unique IDs with exact ID
alignment, 14,849,413 prompt tokens, and 5,000 output tokens. The plugin
server retained its cache from the 30- and 300-request warmup runs; that warmup
is at most 330 earlier requests, and the full run itself contains many
cross-request hits. The baseline had no SimLLM cache.

| Run | Wall time (s) | Throughput (requests/s) | TTFT p50 (s) | TTFT p95 (s) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 962.2118 | 5.1964 | 3.0724 | 3.2660 |
| `tail_anchor`, suffix 512 | 799.2640 | 6.2558 | 2.5581 | 2.8964 |

Throughput improved **20.39%**; median TTFT fell **16.74%** and p95 TTFT fell
**11.32%**. The plugin logged 4,752 semantic reuse hits and 11,104,247
skipped tokens. No SimLLM KV extraction or injection failure was logged.
The full-run margins exceed the 10% target on all three reported metrics.

An independent replay on fresh services with the committed reproduction
scripts produced the following results. Raw paired outputs and the plugin
server log are in [`repeat/`](repeat/); `latency300.json` and
`latency5000.json` contain the machine-checked results. Both checks passed.

| Requests | Policy | Throughput (requests/s) | TTFT p50 (s) | TTFT p95 (s) |
| ---: | --- | ---: | ---: | ---: |
| 300 | Baseline | 5.6311 | 2.7842 | 3.1196 |
| 300 | `tail_anchor`, suffix 512 | 6.7781 | 2.3042 | 2.5794 |
| 5,000 | Baseline | 5.2563 | 3.0401 | 3.2333 |
| 5,000 | `tail_anchor`, suffix 512 | 6.2435 | 2.5650 | 2.8690 |

In this repeat, the full-workload gains were **18.78%** in throughput,
**15.63%** in median TTFT, and **11.27%** in p95 TTFT. Both observations
used baseline card 7 and plugin card 6, so card-specific performance remains
a limitation of this comparison.

The first 5,000-request outputs, server log, and computed aggregates are
retained here. The independent repeat retains both the 300- and 5,000-request
outputs.

## Reproduce on the measured Ascend host

The repository includes the exact synthetic input as
`benchmarks/fixtures/semantic_similar.jsonl.gz`. Its decompressed SHA-256 is
`eecb069a51af828f3388525fcaf84f82cb15a56116fdffc293a69dd77bc9a746`
(5,000 requests). The fixed 30-request warmup and 300-request sample are also
included. The scripts check the input hash, paired IDs and prompt token counts,
then require at least 10% improvement in throughput, TTFT p50 and TTFT p95
for both the sample and the complete replay.

The measured environment used vLLM
`0.28.1.post1.dev143+gf18cf803c.empty`, vLLM Ascend
`0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`, torch-npu `2.13.0rc1`,
CANN 9.1, and the W8A8 model at `/data/jxd/Qwen3.5-35B-A3B-W8A8`.
The model `config.json` SHA-256 is
`5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`;
the quantization description SHA-256 is
`049331a167909d4223126199c7879ca5b58d78fa34439cf99977266081555dde`.
Weights and the compatible Ascend launcher must be present on the test host.

```bash
cd /tmp/simllm-qwen35-adapt
/data/jxd/envs/vllm-hust-v1-cann91/bin/python -m pip install --no-deps .
./benchmarks/start_qwen35_pair.sh /tmp/simllm-replay-servers
PYTHON_BIN=/data/jxd/envs/vllm-hust-v1-cann91/bin/python \
  ./benchmarks/reproduce_qwen35_latency.sh /tmp/simllm-replay-results
```

The start script uses the host's
`/data/jxd/quant-Qwen3.5-35B-A3B/run_swe_server.sh` by default, card 7 for
baseline and card 6 for SimLLM, ports 8107 and 8106. Override the documented
environment variables in the script for other paths or free cards. It refuses
ports with an existing healthy service. The benchmark script expects fresh
services, then runs the same warmup and sequential paired replays. It writes
raw JSONL and machine-readable pass/fail JSON. One full pass takes tens of
minutes; preserve its server logs alongside results. Stop only the services
created by the start script using its PID files after the run.
