# Ascend current-main multi-card evidence campaign

This is the execution contract for
[issue #136](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/136). It prepares the campaign;
it is not a performance result and does not make the issue ready by itself.

## Evidence boundary

Do not calculate or publish a performance percentage until all cells being compared use the same
frozen inputs:

- immutable model revision;
- full vLLM-HUST and vLLM-Ascend-HUST commits;
- image ID, CANN version, torch-npu version, node type, and HCCS/network topology;
- model precision, graph/eager mode, TP/DP/PP/EP, server arguments, and workload arguments.

Readiness logs, historical artifacts, and results from an integration branch remain useful
correctness evidence, but they are not current-main performance points. If a comparable baseline is
missing, mark the cell blocked and publish no delta.

## Required matrix

The dense track is complete only when the following cells have at least three independent service
processes each:

| Workload                 | 1 chip   | 2 chips  | 4 chips  | Load profiles                           |
| ------------------------ | -------- | -------- | -------- | --------------------------------------- |
| random-online            | required | required | required | fixed 1 RPS, matched-load               |
| sharegpt-online          | required | required | required | fixed 1 RPS, matched-load               |
| prefix-repetition-online | required | required | required | fixed 1 RPS, matched-load               |
| agent-research-online    | required | required | required | fixed 1 RPS, matched-load               |
| communication-sensitive  | required | required | required | one explicitly named saturation profile |

Determine each matched-load point with a capacity pilot on the frozen stack. Record the pilot; do
not silently substitute equal low QPS for the matched-load cells.

After the dense anchors are complete, run the MoE specialty track on the same frozen stack:

1. MoE baseline;
1. EPLB disabled and enabled as one matched comparison;
1. LatchMoE disabled and enabled, or a retained blocker;
1. current/latest only when both sides can be reproduced without crossing a commit pair.

Keep eager and graph results in separate comparison scopes. Every targeted pair needs one stable
`CAMPAIGN_COMPARISON_ID`, a `baseline` or `head` role, and three independent services per role.

## Strict repetition launch

`run-campaign-repetitions.sh` launches `run-single-repetition.sh` once per repetition, which in turn
starts and stops one service. Strict mode rejects the same-server `PERFGATE_MEASURED_RUNS=3`
pattern; three client measurements against one live service are not three independent services.

Freeze and verify the inputs before running a cell:

```bash
export CAMPAIGN_REQUIRE_FROZEN_INPUTS=1
export CAMPAIGN_ID=issue-136-current-main/v1
export CAMPAIGN_COVERAGE_CLASS=full-matrix
export CAMPAIGN_POINT_ROLE=checkpoint
export CAMPAIGN_LOAD_PROFILE=fixed-1-rps

export CURRENT_VLLM_HUST_REPO=/workspace/vllm-hust
export CURRENT_VLLM_ASCEND_HUST_REPO=/workspace/vllm-ascend-hust
export CURRENT_GIT_COMMIT=<full-40-character-core-commit>
export CURRENT_PLUGIN_GIT_COMMIT=<full-40-character-backend-commit>
export CURRENT_IMAGE_ID=<full-64-character-image-id-or-sha256-digest>
export CURRENT_MODEL_REVISION=<immutable-40-to-64-character-model-revision>
export CURRENT_CANN_VERSION=<exact-version>
export CURRENT_TORCH_NPU_VERSION=<exact-version>
export CURRENT_TOPOLOGY=<node-and-link-topology-description>

export ASCEND_RT_VISIBLE_DEVICES=<explicit-runtime-visible-device-list>
export ASCEND_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES"
export PERFGATE_WARMUP_RUNS=0
export PERFGATE_MEASURED_RUNS=1

export CAMPAIGN_SUMMARY_FILE=.benchmarks/issue-136/random-fixed-tp2-summary.json
bash scripts/run-campaign-repetitions.sh <frozen-spec.json> \
  --campaign-prefix issue-136-random-fixed-tp2 \
  --repetitions 3
```

For EPLB or LatchMoE pairs, additionally set:

```bash
export CAMPAIGN_COVERAGE_CLASS=targeted-pair
export CAMPAIGN_COMPARISON_ID=<stable-matched-comparison-id>
export CAMPAIGN_POINT_ROLE=baseline  # use head for the enabled side
```

Strict mode fails before model startup when a commit is not full length, a checked-out repository
does not match its declared commit, immutable model/image identity is missing, the formal role is
invalid, fewer than three repetitions are requested, or same-server measurements are configured.

Each repetition's `env-manifest.json` records both declared and observed source commits, image and
model identity, declared and detected runtime versions, topology, visible devices, campaign role,
load profile, and zero-based repeat index. `CAMPAIGN_SUMMARY_FILE` records every attempted artifact
directory and exit code without replacing the raw repetitions. Artifact validation rejects missing
or mismatched detected CANN/torch-npu versions and declared/observed source commits in strict mode.

## Profiler collection

Collect an unprofiled three-service series first. Then run a separately identified profiler cell
with `run-current-ascend-same-spec-msprof.sh`; never mix profiler-affected latency or throughput
into the unprofiled aggregate.

At minimum retain:

- per-rank AI Core/Vector, copy, communication, and visible-idle time;
- HCCL collective type, count, bytes, wait, and compute overlap;
- scheduler/admission/prefill wait, rank batch/token imbalance, and PP bubble when applicable;
- HBM/KV capacity, preemption, and profiler overhead;
- for MoE, per-expert load, all-to-all, placement/rebalance, and graph capture/replay coverage.

Analyze the raw profile outside the runner, preserve the raw `msprof` directory, and link both the
summary and raw profile from the issue.

## Publication gate

Before publishing:

1. validate every artifact with `scripts/validate-run-artifact.sh`;
1. confirm three contiguous independent-service repeat indices for every required series;
1. compute median and IQR while retaining all raw repetitions;
1. confirm every comparison has matching model, hardware, topology, workload, and frozen stack;
1. publish `leaderboard_multi.json` only after the required matrix is complete;
1. link the artifacts, profiler summaries, implementation issues, and any explicit blockers back to
   issue #136.
