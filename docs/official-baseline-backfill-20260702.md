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

The old 2-chip official submission using workload name `sonnet-throughput`
was moved out of canonical submissions and replaced by
`sonnet-throughput-2chip` so it can align with the vLLM-HUST compare scope.

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

