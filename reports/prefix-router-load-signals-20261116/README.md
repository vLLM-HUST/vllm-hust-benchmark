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
zero measured prefix hits. Aggregates do not contain P95/P99, throughput or individual request
distributions and cannot establish tail-latency or throughput guarantees.

The measured experiments used four Ascend NPUs, two TP=2 Qwen3.5-35B-A3B backends, and the
experiment-specific configurations identified by the private manifests and their published hashes.
This record does not claim a rerun on the final PR head, a frozen five-dataset standard evaluation,
or full publication of native runtime evidence. See the PR for the individual study designs,
limitations and explicit request-count/Decode/queue configurations.
