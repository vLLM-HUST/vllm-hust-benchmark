# BetterScale MMLU-Pro and FrontierScience evaluation

Measured on 2026-10-15 with Qwen3.5-35B-A3B BF16. Both full evaluations and separate load tests
completed. MMLU-Pro accuracy is effectively unchanged; FrontierScience outcomes move in different
directions. Both load tests report higher throughput than the earlier website deployments.

This is a single-run, same-client comparison of different engine deployments, not an isolated
optimization ablation, official leaderboard replication, or V5.5 acceptance. No leaderboard entries
or website data are modified by this report. HLE and later multimodal changes are outside scope.

## Task quality

| Metric                                     |   Earlier website run |           BetterScale |
| ------------------------------------------ | --------------------: | --------------------: |
| MMLU-Pro                                   |   9609/12032 (79.86%) |   9611/12032 (79.88%) |
| FrontierScience Olympiad                   |          57/100 (57%) |          63/100 (63%) |
| FrontierScience Research, score at least 7 |          5/60 (8.33%) |          2/60 (3.33%) |
| Research mean over valid grades            | 2.53088/10, 57 grades | 2.62035/10, 57 grades |

MMLU-Pro retains all 12032 questions in the denominator. BetterScale has 65 invalid answers, 1548
length-ended outputs and zero transport failures; the earlier run had 56 invalid answers and 1533
length-ended outputs.

FrontierScience has zero solver or judge transport failures. Olympiad/Research unresolved judges are
4/3, versus 3/3 previously. Solver truncations are 27/9 (previously 33/6); judge truncations are 8/9
(previously 4/7). First-complete-VERDICT extraction remains unchanged, including valid verdicts in
length-ended responses. Unresolved grades remain in the denominator as nonsuccess; missing Research
grades do not become numeric zeroes. No evaluation requests were retried.

The same Qwen model acts as solver and judge. Research threshold successes decreased while its
valid-grade mean increased. This does not establish universal quality preservation or identify which
numerical, generation or judging differences caused a changed result.

## Independent load tests

| Metric                                  | Earlier website run |      BetterScale |
| --------------------------------------- | ------------------: | ---------------: |
| MMLU-Pro output tokens/s                |             442.815 | 624.824 (1.411x) |
| MMLU-Pro mean TTFT, seconds             |              53.476 |           40.627 |
| MMLU-Pro mean TPOT, milliseconds        |              31.917 |           23.617 |
| FrontierScience output tokens/s         |             555.664 | 922.461 (1.660x) |
| FrontierScience mean TTFT, seconds      |              33.955 |           20.375 |
| FrontierScience mean TPOT, milliseconds |              26.663 |           16.207 |

All 200 MMLU-Pro and 160 FrontierScience load requests completed without failure. Both use the
original custom-input selection, output cap 256, infinite arrival rate, temperature 0, seed 0 and
the default server chat template. Input/output token totals are 282265/51200 and 62571/40960,
respectively, matching the earlier load tests. These measurements are separate from full answer and
judge generation; TTFT includes burst queueing.

Two TP2 services shared a four-device Ascend 910B2 allocation with parallel execution explicitly
allowed. MMLU-Pro load testing overlapped FrontierScience judging. FrontierScience load testing ran
after the other pair had been released. No repeated-run variance estimate or isolated-host speedup
is claimed.

## Configuration and input parity

BetterScale source is `31fe12b2ddff70209f78d9dab287aedac9dd4bbd`, using the existing qualified
CANN9.1 runtime with vLLM0.25.1. TP2/MTP2, FULL graphs, E16/R36, 20GiB device state and 8GiB Host
cache per rank, 65536 maximum context and 4096 prefill budget are enabled. The earlier deployment
used a different donor version, EP on, no MTP, APC off, FULL_DECODE_ONLY and prefill budget8192.
Client parity is not a claim that these engine configurations are identical.

MMLU-Pro uses the original adapter and plain-server transport: 5-shot prompts, frozen token IDs,
thinking off, temperature0, seed17, max output2048 and concurrency16. Reconstructed `cases.json` is
byte-identical to the earlier full12032-case file. The200 load-test IDs match the published
selection, and every reconstructed raw prompt re-encodes to its frozen accuracy token IDs.

FrontierScience uses the complete frozen0.3.0 client from its public evidence archive and the
pinned160-task dataset. Solver/judge concurrency is1, max output16384/8192, temperature0, seed17,
thinking off and scoring policy `mmlu_extract_first`. The same service runs solver, judge and load
test sequentially. Services are reused after smoke, not cold-cache measurements.

Both services exited normally and all four selected devices were released. The later multimodal
patch was not merged into these runs.

## Evidence and verification

`report.json` contains aggregate results, configuration and release receipts.
`per-case-outcomes.json.gz` contains all12192 task IDs, outcome flags, grades and hashes of the
private raw records, but no questions or answer text. `private-evidence-index.json` binds the
original runtime and load-result files without publishing their contents. `publication.json` binds
the private archives and original input/client identities.

The original MMLU scorer and FrontierScience client were rerun locally against the full raw records
and reproduced the saved results exactly (`verification.json`). The public verifier checks outcome
counts, summaries and file integrity; independent regrading still requires the omitted raw
responses. SHA-256 values are integrity bindings, not signatures.

```bash
python reports/betterscale-dataset-validation-20261015/verify_public.py
```

`export_public.py PRIVATE_ARCHIVE OUTPUT` regenerates the payload-free projections. It does not
modify original run records. Raw questions, responses, images, environment dumps and credentials are
excluded from publication.

Earlier results and their detailed contracts:
[MMLU-Pro](../mmlu-pro-qwen35-b0-precheck-20261012/README.md),
[FrontierScience](../frontierscience-qwen35-b0-precheck-20261013/README.md).
