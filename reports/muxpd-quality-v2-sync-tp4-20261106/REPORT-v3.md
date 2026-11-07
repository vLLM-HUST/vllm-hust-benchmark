# v3 addendum: corrections to v1/v2, and the redefined #23 acceptance (2026-11-07)

REPORT-v1.md and REPORT-v2.md are unchanged and keep their verdicts. Every QUALITY-V2 FAIL recorded
there stays a FAIL under QUALITY-V2. Issue references in this report without a repository name (#23,
#30, #32, #35) refer to `vLLM-HUST/vllm-ascend-MuxPD-hust`, not to this repository.

## 1. Corrections and omissions in v1/v2

- Checksums. `SHA256SUMS-v1` describes the v1 upload. Two files listed there were changed by the v2
  upload: `README.md` (rewritten for v2) and `data/QUALITY-V2-SYNC-TP4-ROW.md` (its `## Result`
  section was appended after collection). `sha256sum -c SHA256SUMS-v1` therefore reports these two
  files as FAILED on the current tree; this is expected and is not corruption. `SHA256SUMS-v2`
  verifies the v2 tree; `SHA256SUMS-v3` verifies the current tree. No manifest lists itself or a
  later manifest.
- Frozen plan hashes. `data/QUALITY-V2-SYNC-TP4.sha256` and `data/QUALITY-V2-SYNC-TP4-ROW.sha256`
  were taken before collection, so they match only the bytes before the `## Result` heading of each
  file (the leading 4696 and 3180 bytes respectively), not the whole file. Check with
  `head -c 4696 data/QUALITY-V2-SYNC-TP4.md | sha256sum` and
  `head -c 3180 data/QUALITY-V2-SYNC-TP4-ROW.md | sha256sum`.
- Uncertainty intervals omitted from the report text (they are in the score files):
  - v1 part A, raw and group vs native: case bootstrap 95% [-2.0, +3.0] pp; two-level bootstrap 95%
    [-2.00, +3.00] pp; run-level Welch undefined (zero run variance in both arms).
    `data/sv2-score.txt`.
  - v2 ROW vs native: case bootstrap 95% [-3.0, +3.5] pp; two-level bootstrap 95% [-2.50, +4.00] pp,
    8% of resamples below -2 pp; run-level Welch undefined. `data/sv2-row-score.txt`.
  - Because every arm is token-identical across its fresh runs, these intervals reflect case
    sampling only, not run-to-run noise.
- Mixed P/D coverage omitted from v2: both ROW runs covered all 25 batches (actual P layer progress
  and D token retirement in the same step). The native control has no MuxPD receipts by design, so
  its uncovered list is all 25 batches in the score files; that is not a failure.
- v2 called ROW "decode-row-invariant"; at TP4 that was an untested expectation, see section 2.

## 2. Redefined acceptance for #23 (maintainer decision 2026-11-07)

Old standard (QUALITY-V2): answer non-regression against batched native; results above, FAIL. New
standard: with ROW enabled, MuxPD output equals the same-configuration native **single-request**
output at TP2 and TP4, judged on complete output token arrays (200/200), with text and finish-reason
agreement reported. The default mode is not required to match batched native per case or per token.
ROW stays default-off; no ROW performance is measured.

Plan frozen and hashed before any run: `data/QUALITY-ROW-SOLO.md` (`data/QUALITY-ROW-SOLO.sha256`).
Driver `data/run_quality_solo.py` imports the frozen `run_quality.py` unchanged and changes only
admission (one request in flight, asserted every engine step). Controller diff:
`data/run_quality_controlled_solo.diff`. The TP2 text-only evidence (`data/qf-row-check.txt`,
`max_tokens` 2048, no token arrays) is not counted as token evidence; TP2 is recollected with token
arrays.

Status: collection in progress; results are added in a later version of this report.
