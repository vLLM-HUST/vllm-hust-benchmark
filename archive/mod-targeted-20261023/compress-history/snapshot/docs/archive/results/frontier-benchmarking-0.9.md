# Frontier benchmark protocol (0.9.0)

> Historical 0.9 campaign and release validation. Use the
> [benchmark procedure index](../../benchmark/README.md)
> for new measurements.

English | [简体中文](frontier-benchmarking-0.9.zh.md)

For the 0.9 campaign, this protocol replaced the former V4.6 A2/A3 matrix as the leaderboard target. Earlier results remain historical evidence and must not be pooled with these workloads. The 0.9.0 release validation uses the same Frontier protocol with a newly pinned model revision and host stack; the 2026-09-27 AgentX positive result and SWE regression below are historical results on merged source commit `17ffdc7`. These local results are not an official website submission.

For SWE run qualification and any later submission to the official website, follow the [Frontier submission guide](https://github.com/vLLM-HUST/vllm-hust-website/blob/codex/frontier-submission-example/data/examples/frontier-submission/README.md). In particular, run a 60-second C4 protocol check before the separate 900-second measurement, preserve the raw `summary.json` and `config.json`, and map `output_tokens_per_second` to total `metrics.output_tps` (the website divides by chip count). A local run or a point on this plugin's HTML page does not automatically publish an official website point. MTP in SWE must use actual generated tokens and observed acceptance, not the historical AgentX synthetic sampler or fixed acceptance length.

## Frozen workloads

| Cohort | Source | Measured window | Context capacity | Headline metrics |
| --- | --- | ---: | ---: | --- |
| SWE prefix reuse | [vLLM-HUST/swe-prefix-reuse](https://github.com/vLLM-HUST/swe-prefix-reuse) | 900 s | 262,144 | In-window output tokens/s/chip and per-request P90 decode speed |
| AgentX 256k | [vLLM-HUST/agentx-bench](https://github.com/vLLM-HUST/agentx-bench) | 900 s, after its original warmup | 262,144 | Official replay throughput, interactivity, latency, and validity |

The current model is Qwen3.5-35B-A3B BF16 on Ascend 910B2 with TP=2. Qwen3.8-27B BF16 would be a separate future model cohort. Freeze the exact model and tokenizer revisions, prepared-file SHA-256, dataset revision, client commit, host commits, plugin commit, and wheel hash. Both SWE arms must use the **same prepared file**; a changed file hash or token delta is a different workload. Do not filter AgentX sessions, clip output, shorten its original warmup, or alter its DAG and delays.

The 2026-09-27 historical runs used model and tokenizer revision `59d61f3ce65a6d9863b86d2e96597125219dc754`. The 0.9.0 release validation uses `712cf74392b05026a6db2bf213d343747d1f6d45`, the revision also used by an older BetterScale website example. It is a separate measurement and must not reuse that example's model/cohort ID.

### CPU-only workload preparation receipt (2026-09-27)

The locally checked-out SWE client at `6861242dbd9f17b707003191e4200b7752911d7c` compiled its bundled Open-SWE sample with the actual Qwen3.5 tokenizer, `transformers==5.17.0`, thinking enabled, and `--max-context 262144`. The source gzip SHA-256 is `a31abfc7dea176a8b4bf339ff8b104194040f29244577eab9ba4ca951a01739b`; the tokenizer backend/template fingerprint is `3f9ca78537850303ee04bfa6640c020be89723c62f37121c0f27a4c0babc53e0`. The prepared file is `../.benchmarks/swe-qwen35-262144.json` (ignored, not committed), SHA-256 `8105957b4001e7fd21373150fb7b844a9931abe9f88c699b3db191b84a8596d3`. It contains 8 accepted sessions, 360 turns, no rejections, a 140,423-token maximum prompt, and a 141,269-token maximum prompt-plus-output history. Both B0 and B1 must read this exact file and recheck its hash before running. The configured 262,144-token capacity is **not** a claim that this SWE sample reaches 256K.

The local AgentX client at `e0c34de525246c98d14df590f9864d7be7a25075` already holds the pinned `semianalysisai/cc-traces-weka-062126-256k` dataset revision `8fecd2fc56694469f758f0afbbb6335ad3043740` in its isolated cache; its preparation receipt records 393 sessions and fingerprint `0d8fdac271289f80`, reverified offline. Its `smoke` profile measures 900 seconds **after** official primers and 10 additional warmup requests per lane. Recheck the client's offline dataset identity again at execution time. Both local client suites passed (SWE 24 tests, AgentX 18 tests); these are input-preparation and client checks, not server protocol qualification or performance results. See the [machine-readable preparation receipt](../../results/evidence/frontier-workload-preparation-20260927.json).

The pinned AgentX wrapper requires matched SPEED-Bench acceptance evidence and a server-side forced-acceptance setting for speculative decoding. This campaign has no such Qwen3.5/CANN 9.1/MTP2 evidence or force-acceptance adapter, so its AgentX B0/B1 arms must both disable MTP and declare that setting. SWE is a distinct cohort and tests real MTP2 generation and acceptance without a synthetic sampler. Do not compare their absolute scores or silently treat the AgentX run as an MTP2 qualification.

## Pairing and interpretation

1. On authorized idle NPUs, cold-start B0 (compression disabled) and B1 (plugin enabled) with equal hardware, model, client, concurrency, and serving settings. Retain the actual launch commands and hardware allocation; do not edit upstream host source.
2. Run a bounded protocol/cache qualification first, then each 900-second window. SWE requires exact token-ID echo, output budgets, streamed usage, and sticky per-session cache routing. Preserve the upstream AgentX result and every invalidity reason.
3. For both arms, retain the observed maximum prompt length, request errors, server-side prefix hits, preemptions, scheduler commits and per-TP-rank acknowledgements, NPU/HBM samples, exit status, and resource release. A 262,144-token configured limit does not prove that a run reached 256K.
4. Compare equal concurrency within one cohort. Show absolute throughput and P90 decode speed as well as B1/B0 deltas. SWE P90 is the 90th percentile of per-completed-request `(output tokens - 1) / (last token time - first token time)`, not `1/P90(TPOT)`. Do not pool different models, prepared-file hashes, or concurrency points into one speedup.
5. Mark a point as an engineering positive result only if protocol checks pass, there are no hidden failures, commits match acknowledgements on every TP rank, and a headline metric improves. Disclose regressions in other metrics. A single 15-minute observation is neither statistical significance nor the formal one-hour AgentX result.

For an unpooled SWE B0/B1 pair, run `python scripts/kvcompress_frontier_swe_pair.py --baseline PATH_TO_B0_RESULT --candidate PATH_TO_B1_RESULT`. It reads the official `summary.json`/`config.json` without rewriting them, rejects invalid or mismatched arms, and prints the official total-throughput, per-chip throughput, P90 decode-speed, and TTFT P95 mapping plus deltas. Inspect the source reports and server logs as well; this helper does not certify hardware identity, actual rank acknowledgements, or public evidence availability.

For an AgentX pair, run `python scripts/kvcompress_frontier_agentx_pair.py --baseline PATH_TO_B0_RUN_JSON --candidate PATH_TO_B1_RUN_JSON`. It validates both original wrapper records and AIPerf exports, checks cohort and deployment pairing, then maps `output_token_throughput.avg` to total output TPS and `output_token_throughput_per_user.p90` to the official per-request inverse-ITL decode-speed P90. Per-chip output TPS divides by both allocated accelerators. Keep TTFT, ITL, request counts, actual maximum input length, errors, and validity alongside these headline axes; the helper does not turn a 900-second smoke into a formal one-hour submission.

SWE measures a fixed long-conversation serving shape, not SWE task-solving accuracy. AgentX synthetic content does not establish semantic answer quality or real tool success. See the [HTML leaderboard](../../results/benchmark-leaderboard.html) for results and negative history; every new point must link to auditable raw evidence.

The local HTML page now separates **Frontier measured points** from **historical records**, following the official [Frontier view](https://vllm-hust.sage.org.ai/leaderboard-runs.html#frontier). The plotted axes are the official P90 decode speed and output tokens/s/chip. `evidence/frontier-results.json` contains both arms of two positive AgentX 900-second smoke pairs, including the post-merge repeat; the SWE cohort remains without a positive point. Each pair shares a `pair_id`, cohort, concurrency, and chip count, with actual maximum input length, raw run IDs, report hashes, evidence URL, and measurement scope preserved. These are plugin-local leaderboard points, not official website submissions or one-hour AgentX results. Run `python scripts/kvcompress_leaderboard.py` to regenerate both browser bundles and `python scripts/kvcompress_leaderboard.py --check` before publishing. No point is inferred from historical percentage deltas.

The old [V4.6 requirements](../benchmark/kv-compress-test-requirements.md), [A2/A3 procedure](../benchmark/benchmarking.md), and [public long-context experiments](public-long-context-benchmarks.md) are historical archives, no longer release gates.

## 2026-10-08 local 0.9.0 release validation

The release candidate wheel ran Qwen3.5-35B-A3B revision
`712cf74392b05026a6db2bf213d343747d1f6d45` in BF16/TP=2 on 910B2
devices 2 and 7. The matched vLLM and Ascend commits were `ebfcfba` and
`7c8ec86`; the wheel SHA-256 was `c8de9f51...`. The configured context was
262,144 tokens with APC, real MTP2, async scheduling, align, and
FULL_AND_PIECEWISE graphs. Both separate SWE C4/60-second protocol checks and
both 900-second windows were valid, with zero failed requests and identical
prepared-workload SHA-256.

| SWE C4/900-second metric | B0, compression off | B1, compression on | B1 vs B0 |
| --- | ---: | ---: | ---: |
| Total output tokens/s | 153.297 | 149.143 | −2.71% |
| Output tokens/s/chip | 76.648 | 74.572 | −2.71% |
| P90 decode tokens/s | 47.920 | 48.424 | +1.05% |
| TTFT P95, ms | 1,472.57 | 1,610.33 | +9.36% (worse) |
| Requests completed in window | 221 | 218 | — |
| Maximum prompt observed | 52,505 | 52,505 | — |

This pair has a protocol-valid **decode P90 improvement**, with regressions
in total throughput and TTFT P95. The longest observed prompt was 52,505
tokens, so it does not establish performance at an actual 256K prompt. During
the B1 service lifetime, 60 scheduler compression commits each had an
acknowledgement from both TP ranks; real MTP acceptance and server-side prefix
hits were logged. The original reports and logs are retained locally, and the
[machine-readable release record](../../results/evidence/kvcompress-qwen35-20261008-v090-swe-pair.json)
provides their hashes. This is one local engineering pair, not an official
website point or evidence of repeatability.

The same candidate then ran the paired AgentX C4/900-second smoke with MTP
disabled in both arms, as required without matched forced-acceptance evidence.
Both original AIPerf reports were submission-valid, with no invalidity reasons,
65 completed requests per arm, and a maximum observed input of about 91.6K.

| AgentX C4/900-second metric | B0, compression off | B1, compression on | B1 vs B0 |
| --- | ---: | ---: | ---: |
| Total output tokens/s | 33.86587 | 33.86584 | −0.00009% (flat) |
| Output tokens/s/chip | 16.93294 | 16.93292 | −0.00009% (flat) |
| P90 decode tokens/s | 27.709 | 28.024 | +1.14% |
| TTFT P95, ms | 1,819.22 | 1,793.15 | −1.43% (better) |
| ITL P90, ms | 45.779 | 43.226 | −5.58% (better) |

The B1 service recorded 31 scheduler compression commits and 62 worker
acknowledgements, with no service errors or preemptions. A post-window cleanup
timeout was logged after all 66 credits returned; the original report remained
valid. This is a protocol-valid **decode P90 and latency improvement**, while
total throughput is effectively unchanged. See the
[machine-readable AgentX release record](../../results/evidence/kvcompress-qwen35-20261008-v090-agentx-pair.json).

## 2026-09-27 engineering smoke (not a leaderboard result)

On idle 910B2 devices 2/3, the candidate served Qwen3.5-35B-A3B BF16 with
TP=2, a 16,384-token context, `mamba_cache_mode=none`, and APC/MTP/async off.
The plugin-owned Triton GDN output operator differed from its BF16 reference
by at most 0.000244. An 8-token short prompt returned HTTP 200 and 16 output
tokens. One 10,000-token retrieval prompt also returned HTTP 200 with exact
10,000 prompt-token and 128 completion-token usage, but the model spent its
output budget on reasoning text and did not return the target code: **0/1
quality, failed**. Its prompt-token SHA-256 was
`e3e11a53dbe68cf57aa6c3ebbbb773ad7a92573f3ee5370406fb959272938502`.
The first run logged only ERROR, so it could not establish that compression
executed. A replay of the same input at INFO level recorded one scheduler
commit and one acknowledgement from each of TP0 and TP1: 10,000 semantic
tokens became 8,192 physical tokens. Quality remained 0/1. See the
[machine-readable record](../../results/evidence/kvcompress-qwen35-20260927-compat-smoke.json).
This does not cover 262,144 context, APC, MTP2, async,
`FULL_AND_PIECEWISE`, or either official 900-second workload; it has no paired
baseline and is excluded from the leaderboard.

The same model was then restarted with APC and `mamba_cache_mode=align`.
Two identical 10,012-token non-thinking retrieval requests each returned the
correct code `10001337` in 9 output tokens, with one scheduler compression
commit and acknowledgements from TP0 and TP1 per request. Each compacted to
8,192 physical tokens. The second request increased cumulative server-side
prefix hits from 8,192 to 16,384, a delta of 8,192 tokens; both output hashes
matched. See the [APC/align engineering record](../../results/evidence/kvcompress-qwen35-20260927-apc-align-smoke.json).
This short functional smoke does not establish MTP2, async scheduling, graph
mode, or either official 256K workload.

With async scheduling additionally enabled, two serial replays of the same
non-thinking retrieval prompt both answered correctly and the second added
8,192 prefix-hit tokens. Two distinct retrieval requests sent concurrently by
separate clients returned their respective correct codes, `10001337` and
`10009256`. All four requests recorded a scheduler compression commit and two
TP-rank acknowledgements, with no observed cross-request answer corruption.
See the [async/APC/align engineering record](../../results/evidence/kvcompress-qwen35-20260927-async-apc-align-smoke.json).
Concurrent clients alone do not prove overlapping device batches. MTP2, graph
execution, 256K context, and the official 900-second workloads remain untested.

An opt-in MTP2 startup diagnostic subsequently reached service readiness after
the plugin separated the extra MTP attention cache from the ten calibrated
target layers. Its first 10,012-token request returned HTTP 500: both TP ranks
reported `aclnnCausalConv1d` internal errors. At that point the cause was not
isolated. That MTP2 diagnostic **failed**, and default startup remained rejected; the incomplete path is
accessible only with `VLLM_ASCEND_KVCOMPRESS_EXPERIMENTAL_MTP2=1` for diagnosis.
See the [MTP2 diagnostic record](../../results/evidence/kvcompress-qwen35-20260927-mtp2-diagnostic.json).

The failure was subsequently traced to two legacy-ABI details: non-speculative
prefill must use only the target-state column of the 2-D MTP cache table, and
speculative convolution must read the accepted-token offset in an extended
rolling state while computing every draft position. With both fixes confined
to the plugin, a later TP=2/eager/APC/align/async/MTP2 smoke answered a 7,012-
token uncompressed control and three 10,012-token compressed requests correctly.
The three compressed requests each produced one scheduler commit and two TP
acknowledgements at an 8,192-token physical budget. The repeated request and
an independent fact both answered correctly. See the
[MTP2 engineering smoke](../../results/evidence/kvcompress-qwen35-20260927-mtp2-apc-align-smoke.json).
This does not qualify MTP2 for default startup or prove a 256K/900-second
performance result; the earlier failure remains in the historical record.
The original PyTorch 2.10/CANN 9.0.1 environment was stopped by the
synchronized Ascend native source's ABI guard. A workspace-isolated PyTorch
2.13/torch-npu 2.13/CANN 9.1 stack and the host native extension were rebuilt
and verified in their original checkouts. See [environment installation](../../development/environment-installation.md).

## 2026-09-27 official SWE C4/900-second pair

The matched stack ran both arms on the same 910B2 devices 2/3, Qwen3.5 BF16,
TP=2, configured 262,144 context, APC, real MTP2, async scheduling, align and
FULL_AND_PIECEWISE graphs. Both separate C4/60-second protocol checks and both
900-second windows exited successfully with `valid=true`, no aborted or failed
requests, and the identical prepared-file SHA-256. B0 kept the plugin's host
compatibility layer loaded but set its compression threshold to 262,272,
above the server limit; B1 used a 9,216-token threshold and 8,192-token
physical budget.

| Official 900-second metric | B0, compression off | B1, compression on | B1 vs B0 |
| --- | ---: | ---: | ---: |
| Total output tokens/s | 220.322 | 202.954 | −7.88% |
| Output tokens/s/chip | 110.161 | 101.477 | −7.88% |
| P90 decode tokens/s | 71.674 | 70.443 | −1.72% |
| TTFT P95, ms | 1,382.17 | 1,631.72 | +18.05% (worse) |
| Requests completed in window | 318 | 295 | — |
| Maximum prompt observed | 74,706 | 70,940 | — |

This is a **valid protocol run and a performance regression**, not a positive
Frontier point. The highest observed prompt remained below 75K; 262,144 is
configured capacity, not achieved workload length. No preemptions occurred and
the measured C4 cache load was low, so this point does not demonstrate a
memory-pressure benefit. The per-window turn mixes differ, and one pair is
not repeatability evidence. The [machine-readable paired record](../../results/evidence/kvcompress-qwen35-20260927-swe-c4-pair.json)
contains run IDs, raw summary/config hashes and limitations; the complete
client outputs remain in ignored local result directories.

## 2026-09-27 official AgentX C4/900-second pair

The original AgentX wrapper and AIPerf replay completed both separate 900-second
smokes with `submission_valid=true`, no invalidity reasons, and the same pinned
393-session dataset. Both arms used the same Qwen3.5 BF16 TP2 deployment on
910B2 devices 2/3, 262,144 configured context, APC, async, align and
FULL_AND_PIECEWISE graphs. MTP was disabled on **both** arms because this
AgentX protocol requires matched forced-acceptance evidence; SWE separately
tested real MTP2. B0 could not compress below the configured context limit;
B1 used an 8,192-token physical budget.

| Official AIPerf metric | B0, compression off | B1, compression on | B1 vs B0 |
| --- | ---: | ---: | ---: |
| Total output tokens/s | 57.492 | 52.627 | −8.46% |
| Output tokens/s/chip | 28.746 | 26.313 | −8.46% |
| Per-request inverse-ITL P90, tokens/s | 63.687 | 61.701 | −3.12% |
| TTFT P95, ms | 1,585.69 | 1,873.90 | +18.18% (worse) |
| Completed requests | 86 | 82 | — |
| Maximum actual input tokens | 169,166 | 169,165 | — |

This is another **valid protocol run and a performance regression**, not a
positive Frontier point. The longest observed input was about 169K, not 256K;
there were no server preemptions. AIPerf warned that streamed usage lacked
per-request cached-token detail, although server-side prefix-hit counters grew
on both arms. The [machine-readable paired record](../../results/evidence/kvcompress-qwen35-20260927-agentx-c4-pair.json)
retains raw report hashes and caveats. Full original client outputs remain in
ignored local result directories. The single pair does not establish
repeatability or semantic answer quality.

Both official cohorts regressed on this initial candidate. A plugin-owned
partial-RoPE fused scoring path was implemented **after** these paired runs;
it passed a real-NPU numerical smoke and reduced a Qwen3.5-shaped 32,768-token
single-layer scoring microbenchmark from 8.06 ms to 0.80 ms (10.08×).

A fresh matched SWE C4/900-second pair using that fused scorer and the same
1,024-token recompute window was also protocol-valid with zero failures. B0
delivered 219.376 total output tokens/s and 70.781 P90 decode tokens/s; B1
delivered 213.334 and 69.832, respectively: **−2.75% throughput and −1.34%
P90 decode speed**. TTFT P95 was 1,353.46 vs 1,354.44 ms. Thus the kernel
speedup materially narrowed the regression but still did **not** produce a
positive Frontier point. Both arms observed a 73,616-token maximum prompt.
See the [fused-scorer paired record](../../results/evidence/kvcompress-qwen35-20260927-swe-fused-rw1024-pair.json).
A separate 4,096-token recompute-window B1 run was also protocol-valid with
zero failures, but against the same fused-scorer B0 it reached 212.788 output
tokens/s and 68.253 P90 decode tokens/s: **−3.00% throughput and −3.57%
P90 decode speed**. TTFT P95 worsened by 6.58%. The longer window did not
recover a positive point; see its [separate negative record](../../results/evidence/kvcompress-qwen35-20260927-swe-fused-rw4096-pair.json).
An additional batched V3 selection optimization was evaluated with
a new same-source pair. One in-progress B0 window using the first batched-selection
draft was deliberately interrupted after a correctness audit found that
unequal segment quotas needed ordered top-k results. Its original client
`error.json` reports `KeyboardInterrupt` and `valid=false` in the ignored
local result directory; it is excluded from all measured comparisons.
The corrected five-of-ten-layer Qwen3.5 candidate then matched B0 at 4/4
correct answers on a separate 32K deterministic fixture, with four B1
compression commits and eight TP-rank acknowledgements. This is a
[limited engineering quality smoke](../../results/evidence/kvcompress-qwen35-20260927-batched-stride2-quality.json),
not SWE task-solving accuracy or a substitute for the 900-second pair.

That corrected batched-selection/five-scoring-layer candidate also completed
a fresh matched SWE C4/900-second pair with zero failures and the same source
hashes. B0 reached 221.500 total output tokens/s, 72.123 P90 decode tokens/s,
and 1,381.69 ms TTFT P95; B1 reached 214.959, 70.517, and 1,610.66 ms.
Thus throughput fell **2.95%**, decode P90 fell **2.23%**, and TTFT P95 rose
**16.57%**. Both maximum observed prompts were 73,616 tokens. See the
[batched/stride-2 paired record](../../results/evidence/kvcompress-qwen35-20260927-swe-batched-stride2-pair.json).
This still is not a positive Frontier point. The per-request output-budget
distribution suggested that many short generations pay the cost of compression.
A separate same-source candidate therefore raised the minimum requested output
length for compression from 64 to 512. Its valid B1 900-second run (0 failures)
reached 212.142 total output tokens/s, 70.689 P90 decode tokens/s, and
1,474.54 ms TTFT P95 against the same B0: **−4.22%**, **−1.99%**, and
**+6.72%**, respectively. The 60-second precheck and
[4/4 separate quality smoke](../../results/evidence/kvcompress-qwen35-20260927-batched-min512-quality.json)
also passed, but the [min-512 paired record](../../results/evidence/kvcompress-qwen35-20260927-swe-batched-min512-pair.json)
shows that this gate did not recover a positive Frontier result. These
single-run comparisons do not establish repeatability.

The same batched V3/minimum-output-512 candidate then completed a fresh
same-source AgentX pair on devices 2/3. Both original wrapper reports were
`submission_valid=true` with no invalid reasons after the unmodified 393-session
dataset, official warmup, C4, and separate 900-second windows. B0 measured
52.953 total output tokens/s (26.477 per chip), 60.925 P90 decode tokens/s,
and 1,618.93 ms TTFT P95; B1 measured 59.226 (29.613 per chip), 64.162,
and 1,952.38 ms. Output throughput improved **11.85%** and decode P90
improved **5.31%**, while TTFT P95 worsened **20.60%**. The maximum observed
inputs were 169,167 and 169,165 tokens, not 256K. The B1 service logged 57
compression commits and 114 TP-rank acknowledgements including warmup.
See the [AgentX positive paired record](../../results/evidence/kvcompress-qwen35-20260927-agentx-batched-min512-positive-pair.json).
This is one engineering-positive 900-second smoke pair, not proof of
repeatability, semantic answer quality, a one-hour formal result, or an
official website submission. Both AgentX arms disabled MTP under the
forced-acceptance rule; SWE's real-MTP pair remained negative.

## Post-merge retest on commit `17ffdc7`

The development candidate was merged with the then-current remote master
(`ed058fa`) before retesting. Both AgentX arms used the same merged plugin
source, frozen 393-session dataset, original warmup, C4, two Ascend 910B2
devices, and separate 900-second windows. Both original reports were
`submission_valid=true` with no invalid reasons or request errors. AgentX
disabled MTP in both arms under its forced-acceptance rule.

| AgentX official metric | B0 | B1 | B1 vs B0 |
| --- | ---: | ---: | ---: |
| Total output tokens/s | 52.703 | 58.593 | **+11.18%** |
| Output tokens/s/chip | 26.351 | 29.297 | **+11.18%** |
| Per-request inverse-ITL decode P90, tokens/s | 60.381 | 63.562 | **+5.27%** |
| TTFT P95, ms | 1,601.49 | 1,858.66 | +16.06% (worse) |
| Completed requests | 83 | 89 | different closed-loop turn mix |
| Maximum actual input tokens | 169,166 | 169,163 | below 256K |

B1 logged 55 scheduler commits and 110 TP-rank acknowledgements, including
warmup. See the [post-merge AgentX paired evidence](../../results/evidence/kvcompress-qwen35-20260927-agentx-postmerge-positive-pair.json).
The positive throughput/decode direction repeated after the source merge, but
these two single-window pairs used different source commits and do not prove
statistical significance. The 900-second smoke is not AgentX's formal one-hour
result, and its synthetic replay does not assess answer quality.

SWE B0 and B1 each passed a separate 60-second protocol check before their
900-second run. Both full reports were `valid=true`, not aborted, with zero
failed requests and all eight prepared sessions completed. Real MTP2, APC,
async scheduling, `mamba_cache_mode=align`, and FULL_AND_PIECEWISE were active
in both arms; only the plugin compression configuration changed.

| SWE official metric | B0 | B1 | B1 vs B0 |
| --- | ---: | ---: | ---: |
| Total output tokens/s | 222.581 | 216.008 | **−2.95%** |
| Output tokens/s/chip | 111.291 | 108.004 | **−2.95%** |
| P90 decode tokens/s | 72.341 | 71.368 | **−1.34%** |
| TTFT P95, ms | 1,283.61 | 1,477.01 | +15.07% (worse) |
| In-window completed requests | 329 | 315 | different turn mix |
| Maximum actual prompt tokens | 74,706 | 73,616 | below 256K |

B1 logged 99 scheduler commits and 198 TP-rank acknowledgements across its
60-second check and 900-second window. The strict same-source comparison and
raw-report hashes are in the [post-merge SWE negative evidence](../../results/evidence/kvcompress-qwen35-20260927-swe-postmerge-negative-pair.json).
Protocol validity and real MTP acceptance do not make this a positive Frontier
point or measure SWE task success. The prior Qwen2.5 standard-release records
remain historical; publication of 0.8 awaits user confirmation.
