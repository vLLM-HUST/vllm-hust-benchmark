# V5.4 Acceptance Boundary

V5.4 is the active delivery test plan. The repository-controlled
`docs/assets/vLLM-HUST标准交付测试方案_V5.4.docx` is a **semantic mirror** generated with the deterministic
V5.4 BF16 transformation. Its identity is pinned by
`src/vllm_hust_benchmark/data/acceptance_v5_4.json` and validated against
`schemas/acceptance_v5_4.schema.json`.

No canonical local V5.4 binary hash, size or rendered page count was supplied to this container. The
repository mirror is SHA-256 `2423d611fce0cd74be47efcac8e3fb79a472e8304bbfe663dd2127974e1ddd46`,
571064 bytes, and retains a cached 87-page application property because this container cannot
perform the final Word/WPS field, font and pagination update. It must be described as a semantic
mirror, never as byte-identical to the local final document.

## Frozen V5.4 rules

- Qwen3.5-35B-A3B on two Ascend 910B2 devices with TP=2 and BF16 (`bfloat16`) is the mandatory
  Pujiang delivery configuration.
- Only B0 and B1 exist. B1 is the final delivery engine frozen at test start with a signed release
  manifest and OCI RepoDigest; the plan does not predeclare a core, plugin or image ID.
- A3 uses fixed TTFT mean/p95/p99 limits of 30/45/60 seconds and TPOT limits of 40/60/80 ms/token.
  The 30720-token B0 Prefill throughput is diagnostic only and cannot generate a threshold.
- A4 cost is complete lifecycle cost divided by successful output tokens that meet the frozen SLO in
  the 30-minute measurement interval.
- E1, E2-G and E2-D are inactive and `NOT_EXECUTED_OPTIONAL`. They create no target, config ID or
  measurement queue. Activation requires a new test-plan version and a fully frozen contract.
- The five primary datasets are MMLU-Pro, HLE-Verified, SWE-bench-Pro, FrontierScience and
  TerminalBench2.1. All other datasets are supplementary material.
- Contracts reference immutable prompt assets and hashes, tokenizer/chat-template identities and
  input shapes. They do not embed a natural-language system or user prompt.
- Every configuration field is frozen before execution. On-site tuning, automatic fallback, best-of
  parameter selection and post-hoc verdict changes are prohibited.

Validate the declaration and generate the mandatory-only queue projection with:

```bash
PYTHONPATH=src python scripts/validate_acceptance_v5_4.py
```

The V4.6 asset and declaration remain only as superseded historical records. They are not execution
authority for V5.4 and cannot be used to promote a result into the current acceptance program.

## Existing-evidence boundary

The current Qwen3.5 B0 workbook projection and unified Frontier measurements establish BF16 as the
measured precision dependency, so V5.4 freezes the mandatory line as BF16 rather than relabeling the
evidence. Precision is aligned, but the workbook still lacks a signed release manifest, OCI
RepoDigest, complete server argv, complete client contract and effective runtime receipt. Those
fields must be supplied before the measurements can become formal V5.4 acceptance evidence; no value
is guessed or reconstructed after the fact.
