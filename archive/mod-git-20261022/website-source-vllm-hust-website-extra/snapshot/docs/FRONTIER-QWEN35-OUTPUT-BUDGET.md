# Qwen3.5-35B-A3B: Native, BidKV and known-output-budget DLA

This matched campaign adds 15 real observations: Native, BidKV and DLA at C1, C2, C4, C8 and C16.
Each cell ran for 900 seconds. All arms passed qualification, completed without request errors and
released their devices. Results belong to their actual MOD groups; the container is execution
provenance.

DLA here uses **declared exact output budgets**, not a learned length predictor. Its admission
capacity checks executed for every request, but none deferred admission. Neither DLA nor BidKV
selected a preemption victim in these windows. Small throughput differences are observations, not
evidence of a causal or repeatable optimization gain.

## Matched configuration

All three arms use the same output-budget-capable core, Ascend runtime, packages, model files,
compiled workload and worker bridge. The new Native control is kept in its own concurrency series
rather than pooled with earlier controls on different core sources. Native disables both candidate
policies; BidKV enables its original victim selector; DLA enables known-budget admission and its
original victim selector through the declared-budget adapter.

Each arm uses two participating Ascend 910B2 chips (TP2, PP1), from a four-chip container
allocation. Throughput is divided by two serving chips. Configuration retains BF16, 262144-token
context capacity, APC, Mamba `align`, async scheduling, real MTP with two draft tokens,
FULL_AND_PIECEWISE graphs, capture sizes 3/6/12/24/48, 16 server slots, a 4096-token scheduling
budget, and 26038239232 KV bytes per chip.

Each fresh service passes 26 exact marker-retrieval probes and a separate C2/60-second prefix-reuse
check before its five serial measurement windows. These gates are not performance points or general
answer-quality certification. Cache is not reset between windows. This is one observation per cell,
without repeated independent service lifecycles.

## Workload and interpretation

The compiled SWE workload fixes prompt lengths and output budgets. Actual generated token IDs feed
subsequent turns, so this is a closed-loop workload: identical compiled shapes do not imply
identical input content or outputs across runs. Generation uses exact `ignore_eos` budgets. DLA
consults that known remaining budget when checking KV capacity. This is not prediction accuracy
evaluation, and capacity checks do not exclusively reserve future blocks.

Only streamed tokens timestamped inside each 900-second window contribute to throughput. Drain
tokens and drain time are excluded. Policy counters include the window and its drain. The importer
recomputes window counts from raw chunks, checks output token IDs and exact budgets, requires
positive prefix-hit deltas, compares common startup arguments and source identities, and requires
successful qualification and device release for all three arms.

| Arm                | Concurrency | Output token/s/chip | Decode P90 token/s | Request errors |
| ------------------ | ----------: | ------------------: | -----------------: | -------------: |
| Native             |           1 |              45.205 |            105.996 |              0 |
| Native             |           2 |              75.874 |             97.940 |              0 |
| Native             |           4 |             108.473 |             78.217 |              0 |
| Native             |           8 |             146.356 |             59.528 |              0 |
| Native             |          16 |             178.101 |             34.966 |              0 |
| BidKV              |           1 |              46.146 |            106.931 |              0 |
| BidKV              |           2 |              76.917 |             98.287 |              0 |
| BidKV              |           4 |             109.825 |             79.033 |              0 |
| BidKV              |           8 |             148.095 |             57.739 |              0 |
| BidKV              |          16 |             180.313 |             34.549 |              0 |
| DLA (known budget) |           1 |              46.401 |            108.037 |              0 |
| DLA (known budget) |           2 |              76.519 |             98.448 |              0 |
| DLA (known budget) |           4 |             107.818 |             78.558 |              0 |
| DLA (known budget) |           8 |             146.612 |             60.610 |              0 |
| DLA (known budget) |          16 |             179.656 |             34.352 |              0 |

DLA's checks, extended checks and passed checks each equal the total started requests:
149/215/324/436/583 for C1/C2/C4/C8/C16. Deferrals and all preemption counters are zero. BidKV's
policy is enabled, with zero preemption calls. Both candidates therefore retain the `not-exercised`
overall decision-effect status; DLA separately records that its admission checks were exercised. The
chart does not label these differences as measured speedups.

## Sources and evidence

- [Campaign scripts and validated importer](https://github.com/vLLM-HUST/vllm-hust-dev-hub/tree/086f3a3434bba649a415653bf8d135c84c6c25da/scripts/frontier_dla)
- [Common core](https://github.com/vLLM-HUST/vllm-hust/commit/d0f22d2bda562156e4dbf433ce645e1769b4f804)
- [DLA declared-budget adapter](https://github.com/vLLM-HUST/vllm-hust-dla/commit/dc20d0f8ea8d09106f77571e1947b9a2f8702545)
- [BidKV source](https://github.com/vLLM-HUST/vllm-hust-bidkv/commit/a0cba97d9abdc99908e46616db622f0e0099127f)

The common Ascend receipt records base `9bf964cb4b87c8cd0d6852c41a55b3c29711fa95` with explicit
qualified patches and actual file hashes; it is not represented as an unmodified base checkout. The
model is pinned to ModelScope revision `712cf74392b05026a6db2bf213d343747d1f6d45`. All 22 model
files passed verification, including 14 weight shards. Model manifest SHA256:
`6238348a1e071f7802cd6a3e808a6ac36592d83fc765872d3a988368b0b144f6`.

The compiled workload SHA256 is `aa23f49e08a946d94eaab21307e9e015140cc8598adfbd5f7e244bdded7b17d0`;
benchmark revision is `6861242dbd9f17b707003191e4200b7752911d7c`. The published evidence JSON
retains client settings, summary counts, raw-request hashes, qualification results, policy deltas
and source receipts. Full request streams remain in the measurement owner's experiment artifacts.

The final campaign archive, containing receipts, deployed scripts, metadata, launchers and source
manifest, has SHA256 `192c3ce54b94aa70d28758ef09e523273afb574e099a4aa9a9c915a8ea3896f2`. Source
archives are separate: core SHA256
`3d96a16276922446b6a6f1657019f90528c6b7d863a1eac23984c6ed13029d81`, DLA plugin SHA256
`90a20b1d6497179d5fa9af309117efe812e551aea6dac19b0e0de1a44df3738b`. Final independent validation
checked 2667 tracked source files against the frozen manifest, found no changes and confirmed no
remaining owners on the four allocated devices. The owned campaign supervisor was then shut down.
