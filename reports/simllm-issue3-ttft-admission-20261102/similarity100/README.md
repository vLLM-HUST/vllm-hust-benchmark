# Final-workload similarity replay: first 100 requests

This is a **targeted diagnostic**, not a full 5,000-request result or a task-quality score. The
source is `resources/workload/final_workloaddata_v2/semantic_similar.jsonl` from the SimLLM
workspace (SHA-256 `eecb069a51af828f3388525fcaf84f82cb15a56116fdffc293a69dd77bc9a746`). We selected
its first 100 lines without reordering: 20 complete groups of five requests. The exact subset is
[`input-100.jsonl.gz`](input-100.jsonl.gz), whose decompressed SHA-256 is
`ab8a23d4117f2c07eca5c94ff5966e29bcf5b40490b7f8eb78c38a15ceb459ea`.

The test used Qwen3.5-35B-A3B W8A8 on one Ascend 910B3 card (physical card 4), TP1, vLLM
`0.28.1.post1.dev143+gf18cf803c.empty`, vLLM Ascend `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`,
eager execution, native APC off, chunked prefill off, block size 128, `max-num-seqs=4`, temperature
0, thinking disabled, and **one output token**. Each group sends its first request alone, then the
other four concurrently. The baseline and plugin arms used the same installed new wheel, model, data
order, and server settings; only `VLLM_ASCEND_SIMLLM_ENABLED` changed. The plugin arm used the
default `MIN_PROMPT_TOKENS=0`. The wheel SHA-256 is
`98997de3e3ee769803958770f29abe437e12fe435e73fe628749e05dee2a8d5b`. Its model config SHA-256 matches
the prior diagnostic (`5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`); this
replay did not rehash every weight file.

| Metric                                 | Disabled baseline |             Default plugin |
| -------------------------------------- | ----------------: | -------------------------: |
| Completed requests                     |               100 |                        100 |
| Warm requests                          |                80 |                         80 |
| Warm TTFT median                       |          0.8661 s |      0.8532 s (1.5% lower) |
| Warm TTFT mean                         |          0.8106 s |                   0.7773 s |
| Full replay wall time                  |         48.4658 s |                  24.1023 s |
| Full replay throughput                 |      2.0633 req/s |               4.1490 req/s |
| First request latency                  |         25.1171 s |                   1.6601 s |
| Throughput after first request         |      4.2401 req/s | 4.4113 req/s (4.0% higher) |
| Semantic hit events / skipped tokens   |             0 / 0 |               94 / 229,936 |
| One-token output identical to baseline |           100/100 |                     92/100 |

The apparent doubling of full-replay throughput is dominated by the baseline's 25.1-second first
request. Because the harness runs that request serially before the other 99, subtracting its latency
yields the after-first-request rates above. The 4.0% difference and 1.5% warm TTFT difference are
small, from one baseline-first run, and do **not** establish a reproducible 10% improvement on this
dataset. Exact served prompt lengths were 1,695–4,084 tokens.

The plugin log records 18 hits on the first request of a marked group and 91 hits whose source
belongs to a different `group_id`. No hit used a previously hit request as its source, so continuous
approximate source rollover was **not exercised**. Of the 80 warm requests, 73 emitted the same
one-token output as baseline. This dataset has no gold answers; one-token equality is neither
correctness nor long-generation quality.

A third arm with `MIN_PROMPT_TOKENS=8192` was attempted. Its model load stalled on the remote DPC
filesystem at the third weight shard and the service never became ready. Its
[startup log](plugin_floor8192.server.log.gz) and [progress log](progress.log) are retained. The
owned process group was terminated and the port released. Since all measured prompts are shorter
than 8192 tokens, the code would exclude them from matching and snapshots, but this arm has **no
inference result** and is not included in the paired numbers.

## Reproduce and audit

[`run.sh`](run.sh) is the exact executed host script;
[`bench_similarity_workload.py`](bench_similarity_workload.py) is the exact client. The script
expects the uncompressed input at `/tmp/simllm-issue3-similarity/input-100.jsonl` and the recorded
wheel/model paths. Before rerunning, unpack the input, verify
[`REMOTE_SHA256SUMS.txt`](REMOTE_SHA256SUMS.txt), and adjust paths and device only if necessary. It
creates a system-site-packages overlay, installs the pinned wheel, launches one server per arm,
records complete streaming outputs and server logs, and stops each server. The third arm failed at
model loading in this attempt. Run `python ../analyze_similarity.py .` in this directory to
recompute [`analysis.json`](analysis.json) from the first two successful arms. Preserve the arm
order or run reversed-order repeats before claiming a stable speedup.

No native APC comparator or full 5,000-request replay was run here. The earlier
[Issue #3 targeted report](https://github.com/vLLM-HUST/vllm-hust-benchmark/tree/a8fbdae1ff61cb4d0c92652b92a0399c663215e9/reports/simllm-issue3-targeted-20261030)
contains an APC diagnostic on a different 12-request workload; its timings cannot be ranked against
this replay.
