# Evidence index

OPset keeps implementation and evidence together while keeping the PyPI wheel
small. Every optimization must provide a correctness gate, a matched baseline,
raw run identifiers, aggregate statistics, an environment identity, and an
explicit evidence boundary.

| Optimization | Correctness | Operator result | End-to-end result | Evidence |
| --- | --- | --- | --- | --- |
| `persistent-matmul-empty` | 36 shape/dtype cases, 108 launches | ZerosLike share 2.4809% to 0.0836%; device operator time -2.7218% | Three matched rounds on ShareGPT, WildChat and InstructCoder; cold-start outliers retained | [report](optimizations/persistent-matmul-empty.md), [summary JSON](../results/persistent_matmul_empty/summary.json) |
| `linear-swiglu-graph` | 10/10 graph cases plus FP16/BF16 tail coverage | M=1/32/128 latency reduced 15.93%/17.92%/16.79% | Qwen2.5-7B ShareGPT, 3 matched rounds: TPOT -3.82%, output token/s +5.24% | [report](optimizations/linear-swiglu-graph.md), [operator JSON](../results/linear_swiglu_graph/operator), [e2e JSON](../results/linear_swiglu_graph/e2e-summary.json) |

## Evidence policy

- Paths committed to the repository are relative to the repository root.
- Server-local model paths, credentials, PIDs, and ambient service state are
  not portable evidence and are not committed.
- Large msprof/MindStudio databases belong in GitHub Release attachments;
  their manifest and SHA-256 remain in Git.
- A positive operator microbenchmark is not promoted to an end-to-end claim.
- Unsupported source structures or shapes fail closed to the native path.
