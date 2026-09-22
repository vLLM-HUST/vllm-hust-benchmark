# BetterScale / native Qwen3.8-27B TP2: official datasets

## Scope

These are September 22, 2026 measurements of a new dual-Ascend deployment using existing leaderboard
datasets and upstream samplers. They are **dataset-matched, not official fixed-target equivalent**.
They neither replace the historical single-card baseline nor claim its hardware, model, runtime or
request-rate contract. Native and BetterScale both use graph execution. VisionArena is not included:
this deployment qualifies text only.

## Online results

Output tokens per second, arithmetic mean of two independent server repeats; no outlier removal or
best-run selection:

| Dataset / scenario       | Native vLLM | BetterScale | Relative gain |
| ------------------------ | ----------: | ----------: | ------------: |
| ShareGPT online          |     159.691 |     218.268 |        36.68% |
| InstructCoder online     |     198.079 |     234.174 |        18.22% |
| Agent Research online    |     167.448 |     187.114 |        11.74% |
| Prefix Repetition online |     162.617 |     188.973 |        16.21% |
| Random online            |     170.776 |     195.430 |        14.44% |

The order was native / BetterScale / BetterScale / native, on the same physical pair. The last two
arms reverse workload order. All 3,328 timed requests succeeded, all four servers exited
successfully, and all per-request input and output length vectors match between arms and repeats.
Agent Research preserves all 32 original output lengths: 78,633 output tokens, maximum 8,099 per
request. These are performance measurements, not model accuracy evaluations.

Both repetitions are preserved in `repeats.json` and compressed raw result files in each submission.
In particular, native InstructCoder throughput varies 205.59 / 190.57 tokens/s; BetterScale is
234.58 / 233.77. The mean does not erase that variation, and two repeats are not a confidence
interval.

## Offline results

All 12 independent offline runs passed (three workloads, two arms, two repeats). The same ABBA order
and reversed final-half workload order were used.

| Scenario / metric                   | Native vLLM | BetterScale |  Change |
| ----------------------------------- | ----------: | ----------: | ------: |
| ShareGPT output tokens/s            |     159.587 |     222.958 | +39.71% |
| Sonnet output tokens/s              |     176.614 |     199.004 | +12.68% |
| Random batch completion latency, ms |    6947.966 |    5931.602 | -14.63% |

Both arms generated 40,141 output tokens for each ShareGPT repeat and 30,000 for each Sonnet repeat.
Throughput reports output-only tokens, not the upstream JSON's input+output sum. Batch latency is
retained as its own metric; TTFT is null for offline measurements. All repeats and raw logs
accompany submissions.

## Deployment and provenance

- Model: local checkpoint Qwen3.8-27B, BF16, TP2, two Ascend910B2 64GiB devices. The `local:` model
  identity makes no unverified Hugging Face repository claim.
- Native vLLM 0.25.1: `752a3a504485790a2e8491cacbb35c137339ad34`.
- vLLM-Ascend 0.25.1rc1: `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95`.
- BetterScale source: `13d43da74d2a99abd0957414a9ed4ea45a0ce4f1`, with qualified 0.5.1 native
  libraries. This is a source-qualified deployment, **not a claim that the unchanged PyPI wheel
  exposes the measured 32K configuration**.
- Python 3.12.11, PyTorch 2.10.0+cpu, torch-npu 2.10.0.post2, Transformers 5.14.1; package inventory
  accompanies each submission.
- Common: 32,768 context, 8 request slots, 2,048 scheduled-token budget, 6GiB KV per rank, async
  scheduling, AIV, engine seed17, MTP off, CPU binding off.
- Native uses its FULL_AND_PIECEWISE graph path and `TASK_QUEUE_ENABLE=1`. BetterScale uses its FULL
  graph bank, FIA preload and `TASK_QUEUE_ENABLE=0`. Actual configurations and strict resolved-spec
  hashes remain distinct.
- APC is enabled for online serving and offline throughput. Online prefix cache is reset after
  warmup and before each timed workload. Offline random latency disables APC (Mamba cache mode
  `none`) to avoid timing repeated warm prefixes.

A separate capacity check passed cold/warm 8,193, 16,385 and 32,704 input tokens with 32 outputs,
plus eight concurrent 16,385-input requests. This validates the no-MTP 32K path, not eight
simultaneously resident full-32K histories: the chosen KV allocation provides 147,456 tokens, about
4.5 full contexts.

## Workload fidelity and explicit overrides

The inherited official spec and every client override are included in metadata. Online uses
streaming, saturated request rate, concurrency8, client seed0, temperature0 and ignore-EOS. Each
scenario has 200 requests except Agent32. InstructCoder's inherited 2,048-request run is explicitly
changed to 200; the inherited finite request rate is not retained. These are deliberate
performance-oriented deployment measurements, not an unchanged official spec.

- ShareGPT: official `ShareGPT_V3_unfiltered_cleaned_split.json`, unchanged upstream
  filtering/sampling. No new context-limit filtering was introduced.
- InstructCoder: `likaixin/InstructCoder`, train split 108,391 rows, revision
  `6a778a720284d6520b56bd03d5c3070930d41071`; lossless local Parquet with the original dataset name
  supplied to the upstream prompt dispatcher.
- Agent Research: original 32-record EvoScientist workload. The upstream CLI uses
  `--custom-output-len -1` to honor per-record lengths rather than its default256.
  `--skip-chat-template` sends original prompts to the chat endpoint, which templates once. No
  request is shortened to fit the former8K guard.
- Prefix Repetition: 10 prefixes, 3,840 shared +256 varying input tokens, 256 output tokens.
- Random online: 1,024 input /256 output tokens.
- Offline: actual upstream `bench latency` and `bench throughput`, not online measurements relabeled
  as offline. Random latency uses batch8, input1,024, output128, 10 warmups and30 timed iterations.
  ShareGPT and Sonnet throughput use200 prompts and0 warmups. Sonnet is the pinned runtime's
  original text. Offline engine seed17 and fixed6GiB KV replace the generic memory fraction.

## Metric semantics and audit

Online throughput is output tokens/s. Offline throughput JSON from this runtime reports input+output
tokens/s; output-only throughput is instead derived from its logged actual output count divided by
measured elapsed seconds, retaining both raw JSON and stdout. Offline batch completion latency is
`batch_latency_ms`, **not TTFT**. The existing catalog's `tbt_ms` alias represents mean per-request
TPOT, not the raw inter-token-latency distribution. Unmeasured memory, constraints and error-rate
observations stay null rather than invented zeros.

Submission aggregates are generated by the standard producer with arithmetic mean and no outlier
rejection. Every sealed submission includes both raw repeats, configuration, dataset manifest,
runtime inventory and protocol-required checksums. No official verification badge, fixed-target
comparison group or fabricated identical spec hash is claimed. Interrupted,
foreign-owner-contaminated and preflight diagnostic runs are excluded in full.

## Submission evidence

The 16 submission directories under `submissions/` have prefix `betterscale-qwen27-tp2-20260922-`,
followed by `online-` or `offline-`, the scenario and arm (`native` / `betterscale`). Each
directory's `repeats.json` links both raw results; `run_leaderboard.json` contains the arithmetic
mean.

The normal publisher also rechecks pre-existing rows. On this snapshot it excludes five old
single-card logprobs entries missing the fixed target's `gpu_memory_utilization` field. Their
rejection reasons are in `leaderboard-data/snapshots/rejected_superseded_report.json`; their
evidence remains in the unchanged historical snapshot. No old numbers are relabeled as this
experiment, and this submission does not relax their admission rules.
