# V5.5 Acceptance Boundary

V5.5 is the active delivery test plan. The repository-controlled
`docs/assets/vLLM-HUST标准交付测试方案_V5.5.docx` is a **semantic mirror** generated from the V5.4 mirror
with the deterministic V5.5 contract transformation. Its identity is pinned by
`src/vllm_hust_benchmark/data/acceptance_v5_5.json` and validated against
`schemas/acceptance_v5_5.schema.json`.

The separately identified canonical final document is 637379 bytes, 83 pages, with SHA-256
`bc61b65162952c28cde066ee3829aa58f15bf9ed0843d485e8b1a832410489d2`. That binary is not present in
this container. The repository semantic mirror is 571121 bytes with SHA-256
`4f28cc44af069e3f2debd1f703926ba39bb3acee5ea64af899e05192c1a5bb41` and retains a cached 87-page
application property. The two identities are recorded separately; the semantic mirror is not
represented as byte-identical to the canonical final document.

## Frozen V5.5 rules

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
- The 35B B0 configuration and provenance receipt is owned and produced by the release and
  evaluation engineering process. It generates and freezes the signed release manifest, both OCI
  RepoDigests, complete server argv and client contract, and effective/startup receipts from actual
  release artifacts and actual runs. Actual values are neither prefilled nor inferred.

Validate the declaration and generate the mandatory-only queue projection with:

```bash
PYTHONPATH=src python scripts/validate_acceptance_v5_5.py
```

The V4.6 and V5.4 assets and declarations remain superseded historical records. They are not
execution authority for V5.5 and cannot promote a result into the current acceptance program.

## Existing-evidence boundary

The current Qwen3.5 B0 workbook projection and unified Frontier measurements establish BF16 as the
measured precision dependency, so V5.5 freezes the mandatory line as BF16 rather than relabeling the
evidence. Precision is aligned, but the workbook still lacks a signed release manifest, both OCI
RepoDigests, complete server argv, complete client contract and effective/startup receipts. The
release and evaluation engineering process must place the release artifacts and actual-run materials
in one immutable, verified manifest. Every field must generate a unique command and agree with the
effective value before the measurements can become formal V5.5 acceptance evidence.

Rebuild the semantic mirror deterministically with:

```bash
python scripts/build_acceptance_v5_5_docx.py \
  docs/assets/vLLM-HUST标准交付测试方案_V5.4.docx \
  docs/assets/vLLM-HUST标准交付测试方案_V5.5.docx
```
