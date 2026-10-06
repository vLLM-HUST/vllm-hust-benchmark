# Issue 214 vLLM 0.18 three-repeat evidence

This directory contains the nine audited vLLM 0.18.0 + vLLM Ascend 0.18.0
candidate suites used by the PPT-safe summary generator. Every workload has
three explicitly named independent service repetitions; no glob selects data.

The candidate source tuple is:

- vLLM: `bcf2be96120005e9aea171927f85055a6a5c0cf6`
- vLLM Ascend: `e18643f8a4d5bd9990727654318ad069ea0b56e2`

The reference source tuple is:

- vLLM-HUST: `43341b177dbaa8c7f04662f71e885ee7dfe22704`
- vLLM Ascend: `0a46364814eedd3314f04eff3490c3ab422438bd`

The reference report retains only derived arrays and its raw evidence path is
unavailable. Therefore the generated report must keep reference evidence at
grade B, suppress all cross-side deltas, and state that input identity was not
verified across the two sides. Candidate evidence remains independently
checkable at grade A.

`random-latency` deliberately uses `batch_latency_ms` because that is the
candidate's valid offline batch metric. The reference array is labeled
`ttft_ms`, so this row is blocked instead of comparing incompatible metrics.

Agent research, prefix repetition, and VisionArena have high candidate IQR.
Their stability notice must remain visible. The InstructCoder suite is the
clean v4 lane; the excluded older canonical backup is not present in this
directory.

The `canonical/` copies are byte-identical to the selected median repetition.
The `repairs/` directory records the independently validated random-latency
metric selection repair. Its original host-bound checksum list is preserved as
`SHA256SUMS.source-absolute`; the adjacent `SHA256SUMS` is the portable,
self-contained checksum list for the copied repair evidence.
