# Closed PR #7 single-card 910B3 record — quarantined (non-public)

- Entry ID: `6a6b8250-e8de-4eb6-86b5-a611b3211bc0`
- Source: closed, never-merged PR #7 ("chore(leaderboard): add singlecard default official
  submission 20260520"), head commit `68bfc69f4468e34f370a534f392c7f133e72a709`
- Workload: `random-online` (Qwen/Qwen2.5-14B-Instruct, FP16, 1024/256)
- Hardware: Huawei `910B3`, 1 chip
- Engine: vllm-hust `0.17.2rc1.dev450+g289b51ab2.d20260511`
- Core commit: `cee0aff18d987ba6fdd86d5e5fec80a20cfc97eb`
- Plugin commit: `f16e2c1a419430e4f018ad41a5cdd5e6b48d0702` (ref
  `feat/bidkv-victim-selector-item1-2`)
- Reported throughput: `225.3081 tok/s`

## Disposition

**Quarantined / non-public / audit-only.** This record must never be restored to `submissions/`,
`leaderboard-data/snapshots/`, the website mirror, or the HF snapshot root, and must never be
compared against `910B2` results or the retired `v0.11.0` target.

## Why it cannot be admitted

1. It only exists in a closed-unmerged PR; only the summary (`run_leaderboard.json`) and
   `leaderboard_manifest.json` were ever committed, and those bytes never reached `main`.
1. `metadata.verified` is `null` — the `910B3` hardware identity was never independently verified.
1. `same_spec.spec_id` is `official-ascend-jan-2026-v0.11.0-random-online-qwen25-14b-910b3` (a
   retired `v0.11.0` target) while the recorded engine version is `0.17.2rc1...` — a spec/engine
   mismatch.
1. The attached `server.stdout.log` proves the run was served with `enforce_eager: True` (eager
   mode), which is disallowed for formal leaderboard data regardless of the empty `enforce_eager`
   field in the same-spec payload.
1. No `env-manifest.json`, `pip-packages.json`, or original `checksums.sha256` were produced — the
   May 2026 run predates the current evidence/admission requirements.

## Evidence located and attached

The original raw evidence was recovered from the gitignored local run output
`.benchmarks/singlecard-default-official-20260520T103357Z/` and is preserved byte-for-byte here:

- `raw_benchmark_result.json`
- `resolved_same_spec.json`
- `server.stdout.log`

The `server.stdout.log` contains no chip/SOC identifier (no `npu-smi` / `910B3` / device-id line),
confirming that hardware identity cannot be independently established from the retained evidence.

## Rerun path (requires 910B3 hardware)

If this measurement is still relevant, it must be re-run on `910B3` using the current clean runtime
with exact source/package provenance, the current workload contract, an explicit `910B3` spec, and
the full evidence set (checksums, env manifest, raw result, server log, verified hardware identity).
Any accepted rerun is published only as an isolated `910B3` hardware series; `910B2` and `910B3`
remain separate comparison/trend identities.
