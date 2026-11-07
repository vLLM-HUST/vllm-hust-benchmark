# Reorganized MOD current-Host port qualification (2026-11-07)

This report follows the fail-closed preflight in
[`reorganized-mod-retest-20261107`](../reorganized-mod-retest-20261107/README.md). It records the
reviewed disposition and current-Host source qualification of the recovered candidates. It does not
replace the earlier evidence and does not contain performance measurements.

## Fixed Host revisions

- Core: `vLLM-HUST/vllm-hust@c4df37dfcc7f72ca41cd43801a468526ecf99a2b`
- Ascend: `vLLM-HUST/vllm-ascend-hust@7602797593170da4c531f1d8882a7b82dce8a1bc`

The Core revision is Ascend's own `.github/vllm-main-verified.commit`, not an independently selected
newer Core tip. PR `vllm-ascend-hust#76` restores the one-argument KV-pool helper ABI required by
this verified pair; the NPU runs below used the reviewed PR code.

The Ascend revision includes the removal of the private Core KV-planning dependency. The recovered
Native GDN copy compatibility bridge consequently has no current purpose and was retired rather than
ported.

## Canonical MOD revisions

| Family                                  | Fixed revision                             | Source qualification                                               | Candidate disposition                                                                                             |
| --------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| `vllm-hust-runtime-micro-optimizations` | `210038bc250d2558c039f6c0874f403c4d89588e` | 10 tests, Ruff, wheel build, Host contract and TP2 receipts passed | Empty-padding, QSL reuse and residual-scale activate on both ranks; paired performance and quality remain pending |
| `vllm-hust-gdn-optimizations`           | `807d1b0d2295a438ec696fdfdc8c2ebd97b6b448` | 12 tests, Ruff, wheel build, Host contract and TP2 receipts passed | Static decay cache and candidate kernel activate on both ranks; Native delayed remains blocked                    |
| `vllm-ascend-attention-optimizations`   | `bc6bb4abeb05a7bf3d26d44400526d74f1a5f01a` | 8 tests, Ruff, wheel build and corrected six-file contract passed  | Core boundary and persistent-empty retired; linear/SwiGLU graph relocated; PyramidKV remains port-required        |

All packages are inert with an empty selector. Unknown, obsolete, relocated or unqualified selectors
fail before installing a Host hook. The active source ports also reject changed Host bytes before
patching.

## Qualification boundary

These checks establish that the retained code is packaged, default-off, auditable and bound to the
verified source pair. The fresh NPU smoke runs additionally establish two-rank activation and one
successful end-to-end response for the runtime-micro and GDN static-decay candidates. They do
**not** establish quality equivalence or a performance improvement.

The isolated environment used Python 3.12, torch `2.13.0+cpu`, torch-npu `2.13.0rc1`, Triton and
Triton-Ascend `3.6.0`, plus the checksum-pinned FLA NPU `v26.9.1-release` A2/aarch64 wheel. On two
Ascend 910B2 devices, both runtime-micro ranks emitted install and first-exercise receipts; both GDN
ranks prepared 18 immutable decay entries and entered the candidate kernel. Health returned 200 and
both responses were exactly `OK`. Raw logs, responses and the compact machine-readable account are
in [`npu-smoke`](npu-smoke/summary.json).

Before any result can leave `retest_pending`, a fresh compatible environment must produce the
protocol artifacts required by the preceding report:

1. independently launched Native and candidate performance arms in fresh processes;
1. three alternating Native/candidate pairs on verified Ascend 910B2;
1. complete quality comparison and the frozen MMLU-Pro workload;
1. raw performance outputs, manifests and final SHA-256 custody.

The attention family currently has no admitted executable candidate, so it cannot enter performance
measurement until a new current-runner PyramidKV (or another attention mechanism) is implemented and
source-qualified.

## Publication decision

The website must keep all three aggregate cards at `retest_pending`. No legacy percentage,
single-MOD result or combination result is transferred to the new canonical revisions.
Machine-readable status is in [`qualification.json`](qualification.json).
