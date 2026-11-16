# Prefix Router PR #8: aggregate experiment evidence

This folder provides aggregate metrics and SHA-256 inventories for selected experiments discussed in
[Prefix Router PR #8](https://github.com/vLLM-HUST/vllm-hust-prefix-router/pull/8). It contains **no
raw requests, conversations, model outputs, per-request timings, or server logs**. The selected
original records remain on the experiment server and are not included in this publication. Hashes
alone neither make the original records publicly available nor allow independent recalculation; a
reviewer needs access to matching original bytes.

## Contents

- `condition-aggregates.csv`: 63 groups from 426 completed experiment points. Groups combine
  repetitions and mirrored backend directions. Each group has at least six target requests. Means
  are weighted by the number of defined target or background requests, as appropriate.

- `experiment-inventory.json`: planned/completed point counts and hashes of the selected compact
  private archives. These archives are not uploaded here.

- `raw-record-hashes-a.json` and `raw-record-hashes-b.json`: SHA-256 and byte lengths for 3,691
  selected raw files, including result, request, routing and gate records. These hashes describe
  files retained privately; they do not include the large monitoring samples and backend logs
  omitted from the compact private archive.

- `study-metadata.json`: allowlisted configuration, client environment, and source hashes from the
  11 original experiment manifests; these are not backend runtime versions.

- `run-summaries.json`: per-run aggregate metrics for seven ordinary-conversation studies, extracted
  from existing report summaries and linked by source SHA-256. Each run covers at least six target
  requests.

- `SHA256SUMS`: hashes of the files in this publication.

Before aggregation, each point's `result.json`, `requests.jsonl` and `routes.jsonl` was checked
against its `complete.json` digest. The inventory contains 11 studies mentioned in the PR; planned
and completed counts match. Per-request records, unfinished or failed attempts outside these
completed points, telemetry samples, and server logs are not published by this summary.

## Reading the figures

`target_ttft_mean_s`, `target_e2e_mean_s` and `target_tpot_mean_s` are arithmetic means across
requests for which the metric is defined. `target_tpot_defined` reports the number of observed
client-token TPOT values; it is not necessarily the total request count. `target_ttft_over_3s` is
the number of target requests with TTFT above three seconds. Background E2E is averaged separately.
`aggregate_cache_hits` is the sum of run-level backend cache counters and includes background
traffic; it is not a per-target count.

In the queue-boundary study, `queued` and `no_waiter` are different background loads; compare
weights within a condition. Controlled prefix-reuse scenarios were built from real text through
concatenation, slicing, warmup, and gated background traffic. Ordinary ShareGPT conversations were
replayed with fixed output lengths. These are independent replays, not same-state counterfactuals,
and do not establish natural-traffic frequency or absence of regressions. Some ordinary studies had
zero measured prefix hits. Apart from the pooled P95 for the ordinary C8 study in
`run-summaries.json`, the aggregates do not contain P95/P99, throughput or individual request
distributions and cannot establish tail-latency or throughput guarantees.

The measured experiments used four Ascend NPUs, two TP=2 Qwen3.5-35B-A3B backends, and the
experiment-specific configurations identified by the private manifests and their published hashes.
This record does not claim a rerun on the final PR head, a frozen five-dataset standard evaluation,
or full publication of native runtime evidence. See the PR for the individual study designs,
limitations and explicit request-count/Decode/queue configurations.

## Plugin experiment record

This is an exploratory Prefix Router plugin study, not an official leaderboard submission. The
comparison is between Lifecycle routing configurations: the previous request-count default, the
existing active-request cost set to 4, and separate opt-in Decode or Router connection-pool queue
signals. The ordinary conversation replays and the deliberately constructed high-reuse contention
cases answer different questions; they should not be combined into a single performance claim.

The public PR describes the workload construction, concurrency, output lengths, repetitions,
weights, selected results, and limitations of each reported study. The condition aggregates provide
counts and mean values; individual request traces, complete per-run configurations, and raw server
logs are not part of this public record. The exact NPU model, precision, backend runtime versions,
per-study Git revision and run timestamps are not established by this public summary. Manifest
source-file hashes identify recorded Router files without asserting a Git revision. Do not infer
them from the machine name or the final Router PR revision. They require verification against the
original run artifacts before being asserted for those earlier studies. The PR's P95 examples
likewise cannot be independently recalculated from the published condition means and hashes alone.

## SWE C4 load-signal diagnostic

`swe-c4-ablation.json` records eight completed windows from a SWE trace replay on four Ascend NPUs
and two TP=2 Qwen3.5-35B-A3B BF16 backends using the final Router PR code. Four arms each ran two
600-second windows at concurrency 4: A uses active request / Decode / Router queue weights 0/0/0, B
uses 4/0/0, C uses 4/1/0 and D uses 4/0/1. Each backend's Router pool was constrained to two
connections. The input was prepared from the bundled eight-trace sample of NVIDIA Open-SWE-Traces;
four trajectories began in each window. The client omitted `cache_salt` because this Router version
falls back on salted inputs. Cross-session cache reuse is therefore possible, so this diagnostic is
not directly equivalent to the salted SWE experiment.

All 256 requests succeeded with zero fingerprint fallbacks. Each window started 32 requests, 28
completed in the timed window and four drained afterwards. Backend cache counters reported 141,312
hits over 210,970 queried tokens per window (66.98%); these are token counters, not the fraction of
requests with hits. A/B/C/D two-run mean TTFT over all 32 requests was approximately
0.996/1.005/0.998/0.995 seconds. Output throughput within the 600-second windows was approximately
17.67/17.44/17.44/17.61 tokens/s. Two short runs do not establish statistical significance or
guarantee non-regression.

The queue signal was zero at every recorded decision. The Decode signal was nonzero in 28 of 32
decisions per run, but removing its cost from the recorded C-arm scores never made the selected
candidate strictly more expensive than the best candidate; the same snapshot check held for the
request cost in B. Ties and future state changes are not resolved by this check, and independent
replays are not same-state counterfactuals. The source request, timing and routing records remain on
the server; the JSON includes their SHA-256 values but not their contents.
