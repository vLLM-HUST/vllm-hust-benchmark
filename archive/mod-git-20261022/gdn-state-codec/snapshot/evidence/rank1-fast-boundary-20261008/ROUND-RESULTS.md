# Rank-one fast-boundary round — 2026-10-08

Evidence label: `real-shadow-pass-online-negative`. This round establishes a
large component improvement and a real-model exactness pass. It does not
establish an online end-to-end speedup.

## Reproducibility boundary

- StateAxis source parent: `913853597069ca5684f565b1a0c5d3e49a93fd02`
- candidate source commit: `466e6f12419909012ec9d41436370a0a43c6652f`
- experimental branch: `experiment/leapquant-rank-scan-20261008`
- model: `/models/Qwen3.5-35B-A3B`
- model revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`
- dataset: AgentX mmap cache
  `240b9f1f4bb25cf6b1e4e9e77947dbac`
- conversation: `4b88101f5ed7507b73138a73dfbf55f972de`
- request shape: 328 prompt tokens, 128 output tokens, temperature 0, seed 0
- hardware: 2 x Ascend 910B2, TP2 for online; one device for component
- software: CANN 9.1.0, vLLM 0.23-derived StateAxis runtime, eager mode
- component shape: 16 TP-local states, 82 steps, five p16 boundaries

The implementation exposes `fitted_ranks` through the fail-closed experiment
configuration. Exact-SVD initialization subtracts only the selected active
ranks, zero-pads the fixed rank-four ABI, and records the initial active rank.

## Gates

| candidate | component batched ratio | component max RMS | real shadow | online result |
| --- | ---: | ---: | --- | --- |
| rank-1, one power iteration | 0.92250x | 0.92597% | not rerun | diagnostic only |
| rank-1, fast boundary | 0.43802x | 1.03367% | pass, max 1.33445% | fail, 1.06714x |
| quant-only, fast boundary | 0.39909x | 1.01149% | fail at layer 18 step 70, 2.39336% | excluded |

The rank-one fast boundary completed all 128 real-shadow decode steps and
improved on the prior rank-four fast boundary, which had failed at step 37.
Its replace-mode output was stable across repetitions, but differed from the
native output as expected for the bounded approximate state.

Matched replace-mode times were `30.299`, `29.824`, and `30.239` seconds before
the logging diagnostic, then `31.128`, `31.015`, `30.680`, `30.436`, and
`30.612` seconds after it. Native times were `28.750`, `28.901`, and `28.568`
seconds. Removing per-boundary INFO logging did not remove the tail regression,
so logging is excluded as the cause. Median ITL was similar (`222.855 ms`
candidate versus `222.151 ms` native), localizing the remaining loss to sparse
boundary/tail events rather than the steady token path.

The previously qualified v68 result remains bounded to its different AgentX
first turn (649 prompt/128 output): its clean-main matched retest measured a
1.16% median online improvement, while bootstrap intervals overlapped zero.
A diagnostic rerun of the new rank-one fast boundary on that first turn had a
`30.2787 s` median total time and `219.830 ms` median ITL over five measured
repetitions. Against the historical native medians (`28.9434 s`, `224.025 ms`)
this is 4.61% slower in total time despite 1.87% lower median ITL. The rerun
used `max_model_len=1024` rather than the historical `2048`, so it is not
promoted as a matched comparison; it is retained as diagnostic negative
evidence. The v68 result must not be transferred either to the harder
conversation or to this different LeapQuant configuration.

## Architecture-selective quant-only follow-up

A full report-only shadow scan of the harder turn identified two sensitive
Qwen3.5 linear-attention layers. Layer 18 reached relative RMS `0.024046777`
and layer 34 reached `0.022272553`; the other 28 linear-attention layers stayed
below the unchanged 2% gate. Report-only behavior is explicitly restricted to
shadow diagnostics; fail-fast remains the default and replace mode cannot
disable it.

Leaving layers 18 and 34 native repaired the observed component correctness
boundary, but did not pass the online no-harm gate:

| arm | runs | total median | ITL median | TTFT median |
| --- | ---: | ---: | ---: | ---: |
| quant-only except layers 18/34 | 5 | 28.5962 s | 221.430 ms | 0.3616 s |
| same-round native | 5 | 28.1815 s | 217.568 ms | 0.3443 s |

The selective candidate was 1.47% slower in total time and 1.78% slower in
ITL. Deterministic 20,000-resample intervals were `[+0.743%, +3.164%]` for
total time and `[+0.777%, +2.663%]` for ITL, entirely on the regression side.
Both arms completed all runs with stable per-arm output hashes. A slot-owned
output-buffer reuse experiment was also negative (`29.0463 s` median, 3.07%
slower than native) and was reverted.

This follow-up narrows the bottleneck to per-layer host dispatch/raw-ctypes
overhead plus boundary tails. Layer exclusion is retained as sensitivity-scan
tooling, not as a promoted performance configuration.

## Decision

Keep the rank-one fast boundary disabled. Preserve it as the leading exactness
candidate because it passes both the large component gate and the harder
real-shadow gate. The next optimization target is cross-layer boundary tail
latency; a bitmap is not justified for the current full 16-record windows,
which contain no structured inactive records. Quant-only is rejected for this
harder workload, and its failure is retained. The previously qualified v68
result remains the bounded positive configuration for its original
649-prompt/128-output first turn only.

Raw local evidence is archived at
`/root/stateaxis-evidence/leapquant/2026-10-08-rank1-fast-boundary/` with a
SHA-256 manifest. The public repository records bounded summaries rather than
machine-specific server logs or private paths as performance claims.
