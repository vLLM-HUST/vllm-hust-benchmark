# Reorganized MOD current-Host port qualification (2026-11-07)

This report follows the fail-closed preflight in
[`reorganized-mod-retest-20261107`](../reorganized-mod-retest-20261107/README.md). It records the
reviewed disposition and current-Host source qualification of the recovered candidates. It does not
replace the earlier evidence and does not contain performance measurements.

## Fixed Host revisions

- Core: `vLLM-HUST/vllm-hust@88ddd5c7e2c2f72c548ff1dcd2bc8040834a9207`
- Ascend: `vLLM-HUST/vllm-ascend-hust@7602797593170da4c531f1d8882a7b82dce8a1bc`

The Ascend revision includes the removal of the private Core KV-planning dependency. The recovered
Native GDN copy compatibility bridge consequently has no current purpose and was retired rather than
ported.

## Canonical MOD revisions

| Family                                  | Fixed revision                             | Source qualification                                                | Candidate disposition                                                                                                             |
| --------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `vllm-hust-runtime-micro-optimizations` | `b2bc63ef2e461675613cd87a20040dff2a891a34` | 10 tests, Ruff, wheel build and four-file Host contract passed      | Empty-padding, QSL reuse and Gemma residual-scale cache ported default-off; RoPE repair retired as absorbed                       |
| `vllm-hust-gdn-optimizations`           | `8e08c5b05a9e7e38d38952400f921caf223c0ca6` | 11 tests, Ruff, wheel build and four-file Host contract passed      | Static decay ported default-off; Native delayed blocked pending runner/lifecycle rewrite; old copy bridge retired                 |
| `vllm-ascend-attention-optimizations`   | `8f1f1f2f7cb928a1293977b80bc80066dd2234f3` | 8 tests, Ruff, wheel build and six-file Core/Ascend contract passed | Core boundary and persistent-empty retired; linear/SwiGLU graph relocated to the operator family; PyramidKV remains port-required |

All packages are inert with an empty selector. Unknown, obsolete, relocated or unqualified selectors
fail before installing a Host hook. The active source ports also reject changed Host bytes before
patching.

## Qualification boundary

These checks establish that the retained code is packaged, default-off, auditable and bound to the
current source revisions. They do **not** establish that a mechanism activates on both ranks,
preserves model output or improves performance on Ascend 910B2.

Formal execution has not started. The installed shared runtime is an older vLLM 0.23 environment and
is not the frozen dependency line for the current Core/Ascend source revisions. It must not be
upgraded in place or used to publish a mixed-version result.

Before any result can leave `retest_pending`, a fresh compatible environment must produce the
protocol artifacts required by the preceding report:

1. exact Core, Ascend and canonical MOD commit identities;
1. Native and independently enabled candidate arms in fresh processes;
1. activation receipts and mechanism counters from both ranks;
1. three alternating Native/candidate pairs on verified Ascend 910B2;
1. complete quality comparison and the frozen MMLU-Pro workload;
1. raw outputs, process logs, manifests and SHA-256 custody.

The attention family currently has no admitted executable candidate, so it cannot enter performance
measurement until a new current-runner PyramidKV (or another attention mechanism) is implemented and
source-qualified.

## Publication decision

The website must keep all three aggregate cards at `retest_pending`. No legacy percentage,
single-MOD result or combination result is transferred to the new canonical revisions.
Machine-readable status is in [`qualification.json`](qualification.json).
