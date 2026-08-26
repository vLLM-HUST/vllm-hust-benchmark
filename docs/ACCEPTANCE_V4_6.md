# V4.6 Acceptance Boundary

The local `vLLM-HUST标准交付测试方案_V4.6.pdf` is the authority for project delivery acceptance. The
machine-readable declaration is `src/vllm_hust_benchmark/data/acceptance_v4_6.json`. The PDF
identity is pinned by SHA-256 in that declaration so a replacement document cannot be silently
treated as the same acceptance version.

## Test tracks

| Track                  | Purpose                                                              | Acceptance authority |
| ---------------------- | -------------------------------------------------------------------- | -------------------- |
| Pull-request CI        | Correctness, schema, static checks, and artifact-contract tests      | No                   |
| Engineering health     | Legacy nightly, public leaderboard, and specialty workloads          | No                   |
| V4.6 formal acceptance | A1-A4 on the frozen B0/B1 contract through the independent evaluator | Yes                  |

Legacy `random-online`, `sharegpt-online`, and the other public workloads remain useful engineering
signals. They must not produce an A1-A4 completion claim. The generated official-target registry is
therefore separate from the V4.6 acceptance declaration.

## CI policy

- The legacy Core and Plugin performance-request workflows are removed. Engineering-health scripts
  remain available for local analysis, but no GitHub Actions request path is active.
- Ordinary pull requests use hosted correctness CI only; they do not occupy NPU resources or submit
  performance evaluations.
- Performance labels and release/formal requests may enter the independent evaluator only after its
  authentication, idempotency, contract, resource, and evidence preflight is implemented and
  admitted. Until then, no automated performance-request path is active.
- The old performance `Merge Gate` workflow is removed. Its CLI and tests remain available for
  developer analysis, but it is not a required check.
- The legacy Hugging Face publication workflow is removed. Any future publication path requires
  prior admission and cross-repository snapshot verification.
- Existing Smoke/Regression request workflows were removed because they submitted legacy 14B
  performance targets rather than running correctness checks.

## Formal workflow status

The V4.6 declaration is fail-closed and is not yet executable. Before adding a formal workflow, the
independent evaluator and registry must support A1-A4 unit IDs, frozen B0/B1 identities, three
independent service lifecycles, signed receipts, exclusive-resource evidence, and all unit-specific
metrics. Until those contracts are reviewed and admitted, no existing leaderboard or Perfgate result
may be promoted to V4.6 acceptance.

Repository administrators must remove the deleted Smoke, Regression, and Merge Gate check names from
Rulesets or branch protection before merging the workflow changes.
