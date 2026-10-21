# Evidence index

- Current activation contract: `evidence/ecpa-contract-20261011/RESULT.json`
  (`manager-contract-artifact-binding-non-performance`).
- Frozen real-NPU result: `evidence/ISSUE15_WEIGHT_LOADER_RESULT.md`
  (`matched-real-npu-qwen-cold-warm-scoped-positive`).
- Qwen3.5-35B-A3B TP2 pre-sharded carrier diagnostic:
  `evidence/qwen35-loader-sharded-qualification-20261012/RESULT.json`
  (`matched-real-npu-carrier-diagnostic-not-mod-performance`). The three
  same-host/same-filesystem repetitions reduced median process-to-Ready by
  10.68% and median weight-iterator time by 35.53%, with identical first-output
  hashes. It was not activated through ECPA and is not formal MOD performance.

The Qwen result is a real matched cold/warm startup result with steady-serving
regression gates. The EP8 result validates only the loader, resource bounds and
transactional barrier. Current mock-worker checks validate activation and
identity propagation, not device performance.

The Qwen3.5 carrier cell is a promotion test for TP pre-sharding. Its cache
state and per-launch dirty state were not controlled, so it remains diagnostic
until repeated through the Manifest 0.3 activation path.
