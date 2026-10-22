# Persistent MatMul: remove redundant zero initialization

## Mechanism

The batch-invariant Triton `linear_persistent` kernel fully writes every valid
output element. OPset replaces its output allocation from `torch.zeros` to
`torch.empty`, avoiding a separate zero-fill kernel without changing the
ordinary ACLNN MatMul path.

## Correctness

- 36 shape/dtype cases and 108 launches.
- FP16, BF16 and FP32.
- Decode/prefill-like shapes and non-divisible M/N/K tails.
- Finite-value checks, exact repeat determinism and native-linear comparison.

## Operator profiling

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| ZerosLike calls | 59,365 | 1,057 | -98.22% |
| ZerosLike time | 215,987.24 us | 7,078.44 us | -96.72% |
| ZerosLike device share | 2.4809% | 0.0836% | -2.3973 pp |
| Total device operator time | 8,706,061.26 us | 8,469,099.88 us | -2.7218% |

Three 64-request matched rounds were recorded for ShareGPT, WildChat and
InstructCoder. Their run IDs, means, variance and standard deviation are in
[`results/persistent_matmul_empty/summary.json`](../../results/persistent_matmul_empty/summary.json).
The first baseline run in each dataset is a visible cold-start outlier and is
retained; the unqualified three-run means must not be presented as a causal
production estimate.

## Environment identity

- Ascend 910B2; CANN 9.1.0-beta.3; ATB 9.1.T6.B010.
- Torch 2.10.0; torch_npu 2.10.0.post4; Triton Ascend 3.2.2.
- vLLM `ba07e4a48fc951300d97eb506217dd530583dea3`.
- vLLM-Ascend baseline `588cdbdc577aa2c2d2e1e0ab362e39f6913d02e3`.
- Triton Ascend `a4e5e4b53959de21563754641f2193eea067994e`.

## Boundary

This optimization applies only when the verified persistent Triton kernel is
selected. Unknown source structures are rejected, and no claim is made for
ordinary ACLNN MatMul or unrelated linear implementations.
