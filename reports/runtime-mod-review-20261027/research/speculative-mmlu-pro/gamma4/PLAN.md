# Native gamma4 versus vSpec candidate — frozen before new performance runs

2026-10-27. Fletcher explicitly requested fixing baselinegamma4 and checking remaining usefulness.
Prior corrected-norm adaptive experiment was stopped; both arms here use original host norm. No
full280/fullMMLU expansion.

## Contract

- Qwen3.5-35B-A3B, same frozen56 prompts/order, original14×16 warmup, temp0/top_p1/max4000, TP2+EP
  BF16 B16 context16384 maxbatched8192 memory0.9 block128;
  APC/async/chunkedprefill/FULL_AND_PIECEWISE as before.
- Native baseline: ascend plugin only, fixed MTPgamma4. Requested capture list equals original
  candidate48-item list; native normalizes it to its width5. Record resolved descriptor sets per
  rank after warmup and after generation.
- Candidate: original frozen adaptive config/environment unchanged
  (max6,candidates2/4/6,initial4,cohort-refill enabled,runtime-switch unset). Verify summary
  gamma_switches0 and no selected gamma other than4; otherwise stop and do not label it a
  fixed-effective-depth comparison. This preserves the artifact rather than silently disabling its
  controller.
- Both share requested capture list; effective FULL gamma4 descriptors should support the same
  request sizes. Candidate retains other-width support/compatibility/refill; remaining comparison is
  whole-configuration, not a pure controller ablation.
- Fresh engines, three repeats each, order native/candidate, candidate/native, native/candidate.
  Same physical0/1, launcher mapping/locks/placement guard. Foreign workloads currently on2–5:
  preserve them, record shared-host condition; do not combine old exclusive-run times with this
  comparison.
- No synchronous logits/sampler/norm probes. Named metadata-only RPC before/after timed region
  records graph configuration/proposer identity. Timing excludes load/compile/capture/warmup/RPC,
  includesgenerate+one JSONL write per completed request, as original protocol.
- Report elapsed wall, outputtokens/s, output lengths, scores/truncation, acceptance, preemptions,
  per-repeat speed ratios. No equal-output kernel speedup or full MMLU accuracy claim.
- Each arm1800sec hard bound; max6 runs, no retry on failure. Stop on nonzero exit, incomplete data,
  degenerate output, preemptions, or observed candidate gamma switch. Shared CPU interference limits
  attribution; positive or negative result preserved.
