# Official v0.18 Baseline Backfill - 2026-07-02

This note records the 2026-07-02 backfill for the public vLLM-HUST
leaderboard. It is intentionally strict: historical data produced with the
old v0.11 baseline, BF16 drift, Qwen3 drift, 910B3 labels, or manual/eagle3
experiments must not be reintroduced.

## Required Comparison Rules

- Preferred pairs must compare `vllm-hust` against official `vllm`, never
  `vllm-hust` against another `vllm-hust` revision.
- Compare scope must match on workload, model, hardware, chip count, node
  count, precision, config type, same-spec id, and setting signature.
- Overview cards must not average across workloads or precision settings.
  Choose the concrete matched sample where `vllm-hust` improves most over the
  official vLLM 0.18 baseline, then display that sample's raw values.
- Machine and precision labels must come from the actual run metadata. Do not
  hardcode 910B2/FP16 except when the artifact actually says so.
- Use `/data` for model/data caches and `HF_ENDPOINT=https://hf-mirror.com`
  for downloads. Do not fill `/home`.

## Added Official Baselines

| workload | engine | chip_count | hardware | precision | throughput_tps | TTFT ms | TBT/TPOT ms | error_rate | same_spec hash |
|---|---|---:|---|---|---:|---:|---:|---:|---|
| agent-research-online | vllm 0.18.0 | 1 | 910B2 | FP16 | 132.51 | 2180.95 | 119.79 | 0.0 | `5af9fe0419bbce7865b546a224fbc6e5089567f2751de5e0171745c215653bec` |
| sonnet-throughput-2chip | vllm 0.18.0 | 2 | 910B2 | FP16 | 3724.28 | n/a | n/a | 0.0 | `378603219b0e520a5afdabeee2f629b4d3b1a517b1ea929a2ca7e95b9da7057c` |
| sonnet-throughput-4chip | vllm 0.18.0 | 4 | 910B2 | FP16 | 4677.00 | n/a | n/a | 0.0 | `9ed51a0a3fceb940baaa120c6addc19edfd9e5a2d91c16b49c0336c59fe7352f` |
| agent-research-online-2chip | vllm 0.18.0 | 2 | 910B2 | FP16 | 132.74 | 294.76 | 112.52 | 0.0 | `68c0e2c40e41e93cff846ec61410e057dcc1886c97825e17af24b5a486a7ae1a` |
| prefix-repetition-online-2chip | vllm 0.18.0 | 2 | 910B2 | FP16 | 218.08 | 344.77 | 112.84 | 0.0 | `c1d04eeac1c9ee6c113430955d59ee368e7b5b617d33b4fea4c09754baf11f46` |
| random-online-2chip | vllm 0.18.0 | 2 | 910B2 | FP16 | 223.80 | 338.34 | 112.72 | 0.0 | `18dde0cb1ccf9d4aa6d005976597b18dc2d0656d6ec1b42dcaa78e0203344455` |
| sharegpt-online-2chip | vllm 0.18.0 | 2 | 910B2 | FP16 | 157.25 | 295.81 | 114.28 | 0.0 | `c887cb407873f7931ec7f392f26d33cd530e6325c51f09fc6c0121a704f6b040` |
| agent-research-online-4chip | vllm 0.18.0 | 4 | 910B2 | FP16 | 121.00 | 319.27 | 122.43 | 0.0 | `6e6c49bde94d8d7f758d1ba4609208f156b598d0ecc614738a6cdfa251d351a2` |
| prefix-repetition-online-4chip | vllm 0.18.0 | 4 | 910B2 | FP16 | 216.17 | 354.62 | 121.75 | 0.0 | `f15ebe6780f7aef9f9f8291db066ab2dcec5004723b81d4275cfbf9f04259e64` |
| random-online-4chip | vllm 0.18.0 | 4 | 910B2 | FP16 | 213.89 | 338.33 | 126.26 | 0.0 | `9ac40d05cabbad50f3ce18aeb6dae80da6e976b1db1bbd11922fe230aec96f4c` |
| sharegpt-online-4chip | vllm 0.18.0 | 4 | 910B2 | FP16 | 157.66 | 307.23 | 117.81 | 0.0 | `edfe4ba54abd2e9b152f215ed951417a38b9ea52a6c118f9692c1e744ea07489` |

The old 2-chip official submission using workload name `sonnet-throughput`
was moved out of canonical submissions and replaced by
`sonnet-throughput-2chip` so it can align with the vLLM-HUST compare scope.

## P0 Multi-Chip Current Backfill

The first managed-dev-hub vLLM-HUST 2-chip current batch was run on 910B2
devices 4 and 6. The matching 4-chip current batch was run on 910B2 devices
0, 1, 2, and 3. Both used
`vllm-hust@ceec19abb0ba590f536d32c8fea6fd569a8ce7ad` and
`vllm-ascend-hust@2db7d065429b936a75c989648c0bdc7f18baba3a`. Each service was
launched through `/home/shuhao/vllm-hust-dev-hub/manage.sh` by
`scripts/backfill_historical_pr_benchmarks.py --managed-dev-hub`; no manual
server command was used.

| workload | engine | chip_count | throughput_tps | TTFT ms | TBT/TPOT ms | same_spec hash |
|---|---|---:|---:|---:|---:|---|
| agent-research-online-2chip | vllm-hust | 2 | 110.83 | 7076.77 | 116.16 | `68c0e2c40e41e93cff846ec61410e057dcc1886c97825e17af24b5a486a7ae1a` |
| prefix-repetition-online-2chip | vllm-hust | 2 | 105.76 | 109945.58 | 141.04 | `c1d04eeac1c9ee6c113430955d59ee368e7b5b617d33b4fea4c09754baf11f46` |
| random-online-2chip | vllm-hust | 2 | 125.29 | 91107.15 | 119.48 | `18dde0cb1ccf9d4aa6d005976597b18dc2d0656d6ec1b42dcaa78e0203344455` |
| sharegpt-online-2chip | vllm-hust | 2 | 120.06 | 40394.04 | 112.60 | `c887cb407873f7931ec7f392f26d33cd530e6325c51f09fc6c0121a704f6b040` |
| agent-research-online-4chip | vllm-hust | 4 | 110.05 | 7210.49 | 117.16 | `6e6c49bde94d8d7f758d1ba4609208f156b598d0ecc614738a6cdfa251d351a2` |
| prefix-repetition-online-4chip | vllm-hust | 4 | 118.26 | 88196.33 | 124.53 | `f15ebe6780f7aef9f9f8291db066ab2dcec5004723b81d4275cfbf9f04259e64` |
| random-online-4chip | vllm-hust | 4 | 128.09 | 83971.91 | 117.13 | `9ac40d05cabbad50f3ce18aeb6dae80da6e976b1db1bbd11922fe230aec96f4c` |
| sharegpt-online-4chip | vllm-hust | 4 | 116.01 | 44291.44 | 115.90 | `edfe4ba54abd2e9b152f215ed951417a38b9ea52a6c118f9692c1e744ea07489` |

All eight P0 2-chip/4-chip online scopes now have exactly matching vLLM and
vLLM-HUST same-spec hashes and can form preferred pairs. The high TTFT values
are preserved as measured; they are not synthetic corrections.

## Coverage After P0 Batch

`leaderboard_multi.json` now has 25 entries:

| workload | chip_count | engines present | throughput points | TTFT points | TBT points | status |
|---|---:|---|---:|---:|---:|---|
| agent-research-online-2chip | 2 | vllm, vllm-hust | 2 | 2 | 2 | comparable |
| prefix-repetition-online-2chip | 2 | vllm, vllm-hust | 2 | 2 | 2 | comparable |
| random-online-2chip | 2 | vllm, vllm-hust | 2 | 2 | 2 | comparable |
| sharegpt-online-2chip | 2 | vllm, vllm-hust | 2 | 2 | 2 | comparable |
| sonnet-throughput-2chip | 2 | vllm, vllm-hust | 4 | 0 | 0 | comparable throughput trend |
| sonnet-throughput-4chip | 4 | vllm, vllm-hust | 4 | 0 | 0 | comparable throughput trend |
| agent-research-online-4chip | 4 | vllm, vllm-hust | 2 | 2 | 2 | comparable |
| prefix-repetition-online-4chip | 4 | vllm, vllm-hust | 2 | 2 | 2 | comparable |
| random-online-4chip | 4 | vllm, vllm-hust | 2 | 2 | 2 | comparable |
| sharegpt-online-4chip | 4 | vllm, vllm-hust | 2 | 2 | 2 | comparable |
| sonnet-throughput | 2 | vllm only | 1 | 0 | 0 | stale retired workload name; do not extend |

`leaderboard_compare.json` has 19 preferred pairs and every preferred pair
contains `vllm` and `vllm-hust`. The newly added 2-chip and 4-chip online
scopes have matching same-spec hashes across both engines.

Multi-node remains uncovered because no official multi-node scenario/spec was
added in this batch. Add a standard multi-node spec first, then run both vLLM
baseline and vLLM-HUST current through the same submission/snapshot path.

## Single-Chip Gap Fill

The follow-up single-chip gap-fill batch added three real-online
`instructcoder-online` historical points. Each service was launched through
`/home/shuhao/vllm-hust-dev-hub/manage.sh`; source worktrees were placed under
`/home/shuhao/.cache/vllm-hust-benchmark-worktrees/...` so the container loaded
the actual historical refs through `/workspace/.cache/...`. Large artifacts and
state stayed under `/data/shared_datasets/vllm-hust-benchmark/...`.

| workload | core ref | plugin ref | throughput_tps | TTFT ms | TBT/TPOT ms | error_rate | same_spec hash |
|---|---|---|---:|---:|---:|---:|---|
| instructcoder-online | `2fb7859dd0` | `51e577b17b` | 158.98 | 32166.16 | 95.51 | 0.0 | `4c318a855603082fb2a6734eb3a210b350c428731f155ebe1539322f32661d9a` |
| instructcoder-online | `dcc06b18f3` | `51e577b17b` | 157.86 | 16609.29 | 93.42 | 0.0005 | `4c318a855603082fb2a6734eb3a210b350c428731f155ebe1539322f32661d9a` |
| instructcoder-online | `ec4847981f` | `51e577b17b` | 159.68 | 7217.66 | 88.40 | 0.0 | `4c318a855603082fb2a6734eb3a210b350c428731f155ebe1539322f32661d9a` |

`instructcoder-online` now has eight throughput-visible points in
`leaderboard_single.json`: the official vLLM 0.18 baseline, the earlier
`2206f1f7b7` point, three existing current/plugin points, and the three added
historical core refs above. The official vLLM baseline still carries the older
resolved spec hash `cd594ec89...`; this hash drift is preserved in the report
and should be eliminated by rerunning the official instructcoder baseline under
the current spec before claiming fully hash-identical comparison for this
workload.

## Operational Notes From This Run

- Official vLLM baseline runs and managed-dev-hub vLLM-HUST runs should be
  serialized even when disjoint NPU devices are available. The official
  admission preflight rejects any concurrent `vllm bench serve` client on the
  host, even if it targets another port/device set. Parallel runs are safe only
  for runners whose admission checks are scoped to their own process group.
- Historical/current backfill follows the same-spec client parameters by
  default. Use `--current-client-temperature 0` only when the official baseline
  spec also includes that client override.
- Keep using `/data/shared_datasets/vllm-hust-benchmark/...` for result roots,
  datasets, model caches, and state files. Do not write large artifacts under
  `$HOME`.
- For managed historical runs, put source worktrees under `/home/shuhao` via
  `--worktree-root`, because the dev-hub container maps `/home/shuhao` to
  `/workspace`. A worktree located only under `/data` is not a valid server
  source path unless the container explicitly mounts it.
- `sonnet-throughput` with chip_count 2 is an old naming residue. New work must
  use `sonnet-throughput-2chip` or another explicit chip-count workload name.

## Runtime Notes

Official `vllm-ascend` v0.18.0 custom op registration aborts in the current
torch/torch-npu 2.9.0 runtime when importing `vllm_ascend_C`. For the official
baseline runner, `SKIP_OFFICIAL_ASCEND_C_EXTENSION_BUILD=1` removes stale
custom-op `.so` files and applies the later upstream sampler fallback: if
`torch.ops._C_ascend.npu_apply_top_k_top_p` is unavailable, sampling falls back
to the PyTorch implementation. This avoids publishing 100% failed agent
baseline artifacts.

The bad `agent-research-online` run with `error_rate=1.0` was not kept in
canonical submissions.

## Historical PR Backfill Plan

Runtime PRs that should be backfilled with real runs:

| repo | PR | merge SHA | reason | suggested workloads |
|---|---:|---|---|---|
| vllm-hust | #54 | `2206f1f7b7` | KV cache admission path | single + multi |
| vllm-hust | #41 | `51621c35bc` | V1 attention prefill/decode boundary hot path | single + multi |
| vllm-hust | #49 | `f273f9c5e2` | split KV tensors/offloading worker | multi, plus single sanity |
| vllm-ascend-hust | #53 | `bf2984e3` | model runner/platform/profiling behavior | single + multi |
| vllm-ascend-hust | #55 | `c00350a3` | speculative decoding API compatibility | spec-decode only |
| vllm-ascend-hust | #66 | `e0686f12` | SimLLM scheduler/KV reuse/KV injection | single + multi, repeated-prefix workloads |
| vllm-ascend-hust | #70 | `312ca80a` | SimLLM robustness and validation | single + multi SimLLM |

Benchmark or metadata PRs that are semantic/tooling nodes, not runtime
performance points:

| repo | PR | merge SHA | handling |
|---|---:|---|---|
| vllm-hust | #55 | `e01005036b` | use as tooling node only |
| vllm-hust | #69 | `ec4847981f` | use as perfgate semantics node only |
| vllm-hust | #77 | `ceec19abb0` | use as same-spec/scenario registry node only |
| vllm-ascend-hust | #63 | `3138d541` | rerun affected history after v0.18/910B2 cleanup |
| vllm-ascend-hust | #68 | `6dbe422b` | rerun affected quantized metadata if used |
| vllm-ascend-hust | #73 | `51e577b1` | use as two-stage perfgate node only |

## Validation Performed

```bash
PYTHONPATH=src python -m vllm_hust_benchmark.cli publish-website \
  --source-dir submissions --output-dir leaderboard-data/snapshots --execute
python scripts/validate_public_leaderboard_snapshots.py \
  --snapshot-dir leaderboard-data/snapshots
PYTHONPATH=src python -m pytest \
  tests/test_official_baselines.py \
  tests/test_official_baseline_matrix_script.py \
  tests/test_prepare_official_env_script.py \
  tests/test_official_runtime_inputs.py \
  tests/test_submission_artifacts.py \
  tests/test_same_spec.py \
  tests/test_cli.py -q
git diff --check
```

Snapshot checks confirmed:

- `leaderboard_single.json` has both `vllm` and `vllm-hust` for
  `agent-research-online`.
- `leaderboard_multi.json` has both `vllm` and `vllm-hust` for
  `sonnet-throughput-2chip` and `sonnet-throughput-4chip`.
- `leaderboard_compare.json` preferred pairs are all `vllm-hust` vs `vllm`.
