# Qwen3.5 kv-materialization arrival-control Frontier evidence

This record covers five real online 900-second windows for `kv-materialization-arrival-control`. It
uses the published `swe-unified-native-20260927` series as the only Native control; no MOD-specific
Native run was created.

## Pinned configuration

- Model: Qwen3.5-35B-A3B, BF16, revision `712cf74392b05026a6db2bf213d343747d1f6d45`; all 27 runtime
  files matched that revision.
- Hardware and topology: 2 participating Ascend 910B2 chips, TP2 / PP1 / DP1, expert parallel
  disabled.
- Runtime base: vLLM `d0f22d2bda562156e4dbf433ce645e1769b4f804`; vLLM-Ascend
  `03766ac696fde5ab1980d80ca0b8543d3580c989`.
- Measured vLLM source: `556671b117716300121b2ac78b112443b83d1958`, adding the generic controller
  seam and telemetry over the pinned base. MOD: `53a6b1130cbff58925782eb62d7e88cd37e4e20a`.
  Extension Manager: `701aa95ae8d5b23b5ea8c8ec475ceaa470393e54`.
- Capacity and scheduling: `max_model_len=262144`, `max_num_seqs=16`, `max_num_batched_tokens=4096`,
  APC enabled, async scheduling enabled, Mamba cache mode `align`, native MTP with two draft tokens,
  thinking enabled, temperature 0, `FULL_AND_PIECEWISE`, and 26,038,239,232 KV-cache bytes per chip.
- Workload: `swe-prefix-reuse/v1`, D1, prepared SHA-256
  `8044561ffa1bb430bea8f778ef814d96649321e1a92654b95f64263b996d5e85`, tokenizer fingerprint
  `3f9ca78537850303ee04bfa6640c020be89723c62f37121c0f27a4c0babc53e0`.

Extension Manager performed inspect, check, plan, configure, enable, launch, status, stop and
release verification. The separate benchmark harness sent the HTTP requests and collected raw token
events.

## Results

| Concurrency | Candidate output tok/s | Native output tok/s |    Gain | P90 decode tok/s/user | Completed | Errors | Controller calls |
| ----------: | ---------------------: | ------------------: | ------: | --------------------: | --------: | -----: | ---------------: |
|          C1 |          90.5911111111 |       91.5455555556 |  -1.04% |        107.3687639186 |       147 |      0 |              148 |
|          C2 |         154.7288888889 |      152.7211111111 |  +1.31% |         98.9256849521 |       222 |      0 |              224 |
|          C4 |         237.5244444444 |      217.4755555556 |  +9.22% |         84.7666614072 |       349 |      0 |              353 |
|          C8 |         325.7511111111 |      291.1066666667 | +11.90% |         59.8154350122 |       474 |      0 |              482 |
|         C16 |         414.2922222222 |      359.7500000000 | +15.16% |         36.7961845774 |       622 |      0 |              638 |

The geometric mean of the five candidate/Native throughput ratios is **+7.13%**. The website
computes this value from the ten raw throughput values; it is not stored as a plugin score.

Every point passed prompt-ID echo, usage, DONE and fixed-output-budget checks. Tokens received only
during the 900-second measurement window count toward throughput; 1, 2, 4, 8 and 16 in-flight
requests respectively were drained afterward and excluded. Every point ended with the service
stopped, the port unavailable, owned processes gone and all four visible physical NPUs released.

## Mechanism evidence

The controller was loaded and invoked 1,845 times across the five windows. It observed 1,745
full-reuse, 30 partial-reuse and 70 recompute decisions. All recompute requests exercised
request-scoped prefix-cache bypass; full reuse exercised anchor-scoped APC. The 30 block-aligned
partial requests were conservatively mapped to anchor-scoped full reuse because the available full
prefix dominated the proposed partial prefix. Scheduler telemetry independently recorded 1,633
partial-reuse and 276 recompute realizations, 75,944 matched cache blocks and 38,883,328 reused
tokens.

The mechanism status is therefore `exercised`, not `not-exercised`. The partial-action fallback is a
real limitation: these windows do not establish a benefit from partial materialization itself,
although both admission-control outcomes and the concrete APC/bypass runtime paths ran.

The machine-readable point summaries and final per-run manifest hashes are in
[`docs/evidence/kv-materialization-qwen35-20260928/summary.json`](./evidence/kv-materialization-qwen35-20260928/summary.json).
An earlier C4 attempt was excluded before publication because the runtime source worktree changed
during its window; none of its measurements were combined with the replacement C4 run.
