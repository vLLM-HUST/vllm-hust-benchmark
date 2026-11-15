# GDN fusion / weight pre-pack evidence (2026-11-17 round)

Tracking issue: vLLM-HUST/vllm-ascend-split-batch-hust#59. Evidence binding:
vllm-ascend-split-batch-hust#61 (calibers) and #63 (round-2 records), main at
`4b4c794dcb5146d78bd419333a6bbedd0db4e5f5`. Full report: [REPORT-v1.md](REPORT-v1.md). SHA256SUMS-v1
covers every file in this directory.

Four quotable conclusions:

- **Output-norm fusion**: original cohort +2.66% is NOT reproduced. Eight interleaved runs give
  +0.44% overall and -0.98% after dropping the cold first run; the fused kernel is proven to
  execute. The figure is kept as history only.
- **GDN split operator group**: -39.7% is a same-window target-group self-ratio (8560 -> 5163 ms);
  the all-kernel-denominator figures are internal only. Service side is at parity (512-in +2.62%,
  2048-in -1.23% after the cold run).
- **Conv weight pre-pack**: serving parity in eager mode (-0.42% over four runs); the original 9.26
  ms local saving is not reproduced and stays historical.
- **Gate fp32 hoist**: local Cast removal is solid; service side is +3.07% over four runs but -0.13%
  once the cold run is dropped, consistent with the recorded within_noise, and the gate stays
  default-off.

Method note: the first server start of a session is 6-12% slower than later starts on this box.
Because the original +2.66% cohort ran off,off,on,on, that bias alone can produce a multi-percent
apparent gain.

## Where to look

- Full report: `REPORT-v1.md`
- Round-2 per-run numbers: `data/output-norm-fusion/abba8-interleaved-16384.json`,
  `data/conv-pack/abba-eager-4runs.json`, `data/split-fusion/abba-{512,2048}in-4runs.json`,
  `data/gate-fp32/abba-random-4runs.json`
- Raw bench output (36 fields each): `data/*/raw/*.json`, coverage notes alongside
- Operator-level replication: `data/split-fusion/l0-replica.json` (23 cases),
  `data/conv-pack/l0-conv-pack.json`, `data/gate-fp32/l0-gate-40layers.json`
- Fused-kernel execution proof: `data/output-norm-fusion/engagement-profile.txt` and
  `data/output-norm-fusion/op-statistic-20261115-8prompt-*.csv`
- Original evidence, linked not copied: `data/output-norm-fusion/ORIGINAL-EVIDENCE-LINKS.md`
- What is missing and why: `REPORT-v1.md` section *Validity and limits*

## File roles

- **Conclusion-bearing**: the five `abba-*` block summaries, `l0-*` replication files,
  `engagement-profile.txt`, and the two 8-prompt op-statistic exports.
- **Supporting**: `non-interleaved-partial.json` (order-effect argument),
  `single-original-commit.json` / `single-batched-16384.json` (configuration finding),
  `abba-default-batched.json` (exploratory).
- **Provenance**: every `raw/*.json` file, `RAW-COVERAGE.txt` notes, `OPSTAT-NOTE.md`,
  `ORIGINAL-EVIDENCE-LINKS.md`.
