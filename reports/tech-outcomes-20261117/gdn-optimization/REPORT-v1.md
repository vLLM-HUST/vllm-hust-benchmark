# GDN fusion / weight pre-pack — reproduction and evidence binding (2026-11-15)

Tracking issue: vLLM-HUST/vllm-ascend-split-batch-hust#59.
Evidence-binding PR: vllm-ascend-split-batch-hust#61, fixed SHA `809d8f71abd4986012767f1e2e3b7573a7960a18`.

**Scope**: four carriers (`gdn_output_norm_fusion`, `gdn_split_fusion`,
`gdn_conv_pack`, `gate_fp32`). No measured value in the plugin repository was
changed; that PR only wrote calibers and boundaries. This report carries the
new measurements and the limits that could not be closed.

## Setup

| item | value |
|---|---|
| model | Qwen3.5-35B-A3B (bf16) |
| stack | vLLM `0.28.1.post1.dev143+gf18cf803c`, vllm-ascend `0.25.1rc2.dev125+hust`, CANN 9.1.0, torch_npu 2.13 |
| device | 2 x Ascend 910B2, TP=2 |
| host fork | `based/split-batch-baseline`: `e59aaeef2` for abba1/abba2; `8499a55aa` for base8499, mbt16k and the main abba3 block (same commit as the original evidence) |
| graph mode | `cudagraph_mode=FULL` |
| env | all `VLLM_HUST_*` gates 0 except the tested one; `VLLM_DISABLE_COMPILE_CACHE=1` |
| client | random dataset, 128 prompts, 512 in / 256 out, concurrency 32 |
| key flag | `--max-num-batched-tokens 16384` (see Results) |

Server command used in this round (authoritative for every number produced
here; executed and validated on this box):

```bash
source /root/.venvs/vllm-ascend-npu.env
python -m vllm.entrypoints.cli.main serve /data/shared_models/Qwen3.5-35B-A3B \
  --served-model-name qwen35 --tensor-parallel-size 2 --max-model-len 8192 \
  --gpu-memory-utilization 0.9 --max-num-batched-tokens 16384 \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL"}'
# gate under test: VLLM_HUST_GDN_OUTPUT_NORM_FUSED={0,1}
```

The original cohort did not record its launch command, so no reproducible
command is claimed for the historical +2.66%. Every figure produced in this
round is reproducible from the command above; it is a method declaration for
the new data, not a claim about how the original run was launched.

Metric definition: `output_throughput` from `vllm bench serve` result JSON
(output tokens / generation window). A/B delta = `mean(on) / mean(off) - 1`.

## Results

### Main block — interleaved, throughput magnitude aligned

Run on `8499a55aa` (the same commit as the original evidence) with
`--max-num-batched-tokens 16384`.

### abba3: off,on,on,off

| run | arm | out tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|
| 1 | off | 741.61 | 1755.5 | 35.78 |
| 2 | on | 791.19 | 1608.6 | 34.27 |
| 3 | on | 798.50 | 1546.5 | 34.14 |
| 4 | off | 791.62 | 1545.6 | 34.50 |

- off mean **766.61** (741.6–791.6, intra-arm spread 6.74%)
- on mean **794.85** (791.2–798.5, intra-arm spread 0.92%)
- all four runs: **+3.68%**
- excluding run 1 (cold start): **+0.41%**

The first server start of a session is systematically ~6% slower; runs 2–4 sit
in a narrow 791–799 band regardless of arm. One interleaved block therefore
cannot separate the treatment from the order effect.

### Configuration finding

| host commit | batched tokens | off throughput | note |
|---|---|---|---|
| `e59aaeef2` | default | 618.97 | 19% below the original cohort |
| `8499a55aa` | default | 620.29 | same magnitude → the host-code change is **not** the cause |
| `8499a55aa` | 16384 | 783.99 | matches the original 760.32 → the batching limit was the cause |

The comparison at default batching shows both commits deliver ~620 tok/s,
while raising the batching limit on the original commit delivers ~784 tok/s.
The 19% gap is therefore a configuration difference, not a code regression.

### Other blocks (exploratory, not evidence)

### abba2 (e59aaeef2, default batched tokens)

| run | arm | out tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|
| 1 | off | 625.59 | 1733.6 | 44.39 |
| 2 | on | 618.24 | 1818.5 | 44.54 |
| 3 | on | 616.10 | 1664.6 | 45.45 |
| 4 | off | 612.35 | 1703.7 | 45.56 |

- mean delta **-0.29%**; both arms inside their own spread.

### abba1 (e59aaeef2, non-interleaved off,on; interrupted after 2 runs)

| run | arm | out tok/s | TTFT ms | TPOT ms |
|---|---|---|---|---|
| 1 | off | 632.44 | 1767.5 | 43.68 |
| 2 | on | 646.38 | 1870.0 | 42.19 |

- first pair delta **+2.20%**, close to the original +2.66%; the original
  evidence used the same non-interleaved `off,off,on,on` order, so a warm-up
  effect placed on the second arm is a plausible confound.

### L0 reproduction

`data/split-fusion/l0-replica.json` (23 cases, NPUGraph packed replay). All
cases bitwise-identical to the stock path. Representative deviations:

| case | evidence | replicated |
|---|---|---|
| unpack M=32 | 6.09 -> 3.39 us | 6.75 -> 3.80 us |
| unpack M=1867 | 18.15 -> 11.67 us | 17.30 -> 11.25 us |
| rearrange M=1867 | 24.62 -> 11.48 us | 23.53 -> 9.68 us |
| norm strided T=16 | 73.82 -> 7.65 us | 71.27 -> 6.56 us |
| norm strided T=128 | 96.94 -> 12.68 us | 94.83 -> 12.33 us |
| gate 40 layers | 1823.11 -> 33.19 us (eager) | 125.09 -> 3.11 us (graph) |

### Fused-kernel engagement

`data/output-norm-fusion/engagement-profile.txt`: `_output_norm_kernel` appears
in 6 files of the on-arm profile and 0 files of the off-arm profile, while
`aclnnAddRmsNorm` appears in 8 files on both arms. The null serving result is
therefore not caused by the gate failing to fire.

## Round 2 — full re-test of all four carriers (2026-11-15)

Every carrier got a fresh ABBA block on `8499a55aa` with
`--max-num-batched-tokens 16384` and per-run raw result files. Each block is
`off, on, on, off`.

| carrier | workload | off tok/s | on tok/s | delta (all 4) | off spread | on spread |
|---|---|---|---|---|---|---|
| output_norm (8 runs) | 512/256 @32 FULL | 782.35 | 785.79 | +0.44% | 2.98% | 2.42% |
| conv_pack (eager) | 512/128 @8 | 27.68 | 27.56 | -0.41% | 1.77% | 3.10% |
| split 512-in | 512/256 @32 FULL | 788.17 | 804.92 | +2.13% | 0.99% | 0.52% |
| split 2048-in | 2048/256 @32 FULL | 598.74 | 628.73 | +5.01% | 13.50% | 3.94% |
| gate_fp32 | 512/256 @16 FULL_PIECEWISE | 518.25 | 534.17 | +3.07% | 6.62% | 2.38% |

### First-run bias is the dominant effect

In four of five blocks the first server start of the session is materially
slower than the same arm's later run:

| block | first run | same arm, later run | first vs later |
|---|---|---|---|
| abba3 (off first) | 741.6 | 791.6 | -6.3% |
| gate_fp32 (off first) | 501.65 | 534.86 | -6.2% |
| split 2048 (off first) | 560.87 | 636.62 | -11.9% |
| split 512 (off first) | 792.06 | 784.28 | +1.0% |
| abba4 (off first) | 771.83 | 793.56 | -2.7% |

Because the ABBA order always starts with `off`, a cold first run
systematically depresses the off mean and inflates every on/off delta. Dropping
the first run of each block changes the verdicts materially:

| carrier | delta, all runs | delta, first run dropped |
|---|---|---|
| output_norm | +0.44% | -0.98% |
| conv_pack | -0.41% | +0.47% |
| split 512-in | +2.13% | +2.63% |
| split 2048-in | +5.01% | -1.24% |
| gate_fp32 | +3.07% | -0.13% |

After the cold run is removed, four of five blocks sit within ±1.3% of parity,
and the split-2048 result (-1.23%) reproduces the original -1.27% almost
exactly. The only block still showing a positive delta is split-512
(+2.63%), which is also the only block whose first run was
not depressed.

The original +2.66% was measured with a non-interleaved `off, off, on, on`
order, i.e. both off runs were exposed to the cold start and both on runs were
not. This round's data shows that such an order alone can produce a
multi-percent apparent gain without any code difference.

## Gains

- `gdn_output_norm_fusion`: original cohort +2.66% (760.32 -> 780.57 tok/s,
  2 runs/arm, non-interleaved). **Not reproduced**: this round's 8-run
  interleaved block gives +0.44% overall and -0.98%
  after dropping the cold run. The original figure is kept as a historical
  record only.
- `gdn_split_fusion`: **-39.7%** = 263k KV cohort target group 8560 -> 5163 ms,
  self-ratio denominator = the same-window baseline target group. Other cohorts
  are 53.5% (66k KV) and 55.3% (pure decode). The all-kernel-denominator
  figures 2.41% / 5.00% / 5.05% describe a share of total kernel time and are
  internal references only.
- `gdn_conv_pack`: local 308.62 -> 0.20 us/layer, 30 layers 9.26 -> 0.006 ms/step
  (eager, `torch.npu.Event`, iters=1000). The local L0 replication gives
  4.48 -> 0.10 us eager and 0.56 -> 0.66 us graph-packed, so the local saving is
  highly caliber-dependent and the original 308 us figure is not reproduced.
- `gate_fp32`: local 40x `Cast` 1823.11 -> 33.19 us (eager); graph-packed
  125.09 -> 3.11 us; in-graph `Cast` count falls by 640 per rank.

## Costs

- `gdn_output_norm_fusion`: no service gain. +0.44% over all runs,
  -0.98% after dropping the cold run; off-arm spread
  2.98% is larger than the effect under test.
- `gdn_split_fusion`: operator-group saving does not reach wall time in the
  historical cohort (+0.46% / -1.27%, both inside 4.18% / 4.29%). In this round
  the 512-in block shows +2.63% and the 2048-in block
  -1.24% after dropping the cold run; with n=2 per arm this is
  not enough to claim a gain.
- `gdn_conv_pack`: serving -0.41% over four eager runs; previous
  single-round data showed 29.60 -> 29.52 tok/s with three metrics pointing in
  different directions. Parity only.
- `gate_fp32`: +3.07% over all four runs is driven by a depressed
  first off run; after dropping it the delta is -0.13%.
  Consistent with the recorded `within_noise`; gate stays default-off.

## Validity and limits

1. The four carriers were measured on different host commits and dates; their
   numbers are not additive. No four-way combined run exists.
2. The original `op_statistic.csv` **file** was never committed and cannot be
   regenerated by re-running. Its structured content (off/on target-group times
   and per-operator counts) is, however, retained in
   `gdn_split_fusion/evidence.json` — see `ORIGINAL-EVIDENCE-LINKS.md`. This is
   why the self-ratio 39.7% is reproducible while the all-kernel-denominator
   figures are not independently recomputable.
3. The original per-request traces are not retained, but the four per-run values
   behind +2.66% (off_1/off_2/on_1/on_2 with six metrics each) are in
   `gdn_output_norm_fusion/evidence.json` under `serving_end_to_end.runs`; linked
   at a fixed SHA rather than re-uploaded.
4. The original cohort did not record its server command; this limits only the
   reproduction of the historical +2.66%, not this round's figures. The command
   in Setup is the authoritative method for all new data, was executed on this
   box, and is not presented as the original command.
5. This round's off-arm absolute throughput (783.99 / 766.61)
   is close to, but not identical with, the original 760.32.
6. No `msprof` timeline and no kernel-bound counter-test were collected, so the
   "kernel saving absorbed by slack" mechanism stays a hypothesis.
7. All four carriers were re-tested in round 2 (one ABBA block each), but each
   arm still has only 2 runs, so the per-arm spread is estimated from two
   points. The split-512 result is the only block showing a positive delta
   after dropping the cold run; it is reported as an observation, not a gain.
8. All four carriers are `default_enabled: false`; none of these figures
   describes default product behaviour.
9. Positive, zero and negative results are all retained, per
   vllm-hust-benchmark#664.
10. Round-1 blocks (abba1/abba2/abba3) reused one bench directory per arm, so
    only the last run of each arm has a raw `vllm bench serve` file there.
    Round-2 blocks write one directory per run, so all four raw files per block
    are retained. See `data/output-norm-fusion/raw/RAW-COVERAGE.txt`.
11. The 128-prompt `op_statistic` export failed: `msprof --export` returned 0
    but produced only `api_statistic.csv` on the 3 GB per-arm profile. The
    archive therefore carries the 8-prompt exports only, which are used to
    confirm that the fused kernel executes, not to recompute any all-kernel
    denominator.
12. The first server start of a session is 6-12% slower than later starts on
    this box. This bias is recorded explicitly in Results because it dominates
    blocks whose `off` arm runs first, which is the order the original +2.66%
    cohort used.

## Artifacts

| path | content |
|---|---|
| `data/output-norm-fusion/abba-interleaved-16384.json` | main 4-run block |
| `data/output-norm-fusion/abba-default-batched.json` | exploratory 4-run block |
| `data/output-norm-fusion/non-interleaved-partial.json` | exploratory, 2/4 runs |
| `data/output-norm-fusion/single-original-commit.json` | `8499a55aa` single arm |
| `data/output-norm-fusion/single-batched-16384.json` | batched-token single arm |
| `data/output-norm-fusion/engagement-profile.txt` | profiler hit counts |
| `data/output-norm-fusion/raw/*.json` + `RAW-COVERAGE.txt` | 8 byte-for-byte `vllm bench serve` results (36 fields each) and their coverage limit |
| `data/output-norm-fusion/op-statistic-20261115-*.csv` + `OPSTAT-NOTE.md` | **new** `msprof` exports for both arms; explicitly not the original file |
| `data/output-norm-fusion/ORIGINAL-EVIDENCE-LINKS.md` | fixed-SHA links to the original per-run values and op_statistic aggregates |
| `data/output-norm-fusion/abba8-interleaved-16384.json` | round-2 8-run output_norm block |
| `data/split-fusion/abba-512in-4runs.json` / `abba-2048in-4runs.json` | round-2 split blocks |
| `data/conv-pack/abba-eager-4runs.json` | round-2 conv_pack eager block |
| `data/gate-fp32/abba-random-4runs.json` | round-2 gate_fp32 block |
| `data/split-fusion/op-statistic-reconstructed-from-evidence.csv` | structured transcription of the retained historical values, labelled reconstructed |
| `data/split-fusion/l0-replica.json` | 23 L0 cases |
| `data/conv-pack/l0-conv-pack.json` | round-2 conv_pack L0 (eager + graph packed) |
| `data/conv-pack/NOTE.md` | why this carrier has no service-side history |
| `data/gate-fp32/NOTE.md` + `l0-gate-40layers.json` | no new serving data, L0 only |

Raw profiler trees (263 MB / 297 MB) and server logs are not uploaded.
`SHA256SUMS-v1` covers every file in this directory except itself.
