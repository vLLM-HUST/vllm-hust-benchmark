# MuxPD #23 TP4 synchronous protocol: QUALITY-V2 records and redefined ROW acceptance

Issue references are to `vLLM-HUST/vllm-ascend-MuxPD-hust` (#23 is its quality issue).

- [REPORT-v1.md](REPORT-v1.md): QUALITY-V2, default synchronous raw recipe FAIL (3 systematic losses
  \> 2).
- [REPORT-v2.md](REPORT-v2.md): QUALITY-V2, opt-in ROW on the same protocol FAIL (5 > 2).
- [REPORT-v3.md](REPORT-v3.md): corrections to v1/v2 (checksums, intervals, coverage) and the
  redefined #23 acceptance: ROW output equal to native single-request output at TP2 and TP4.
- [REPORT-v4.md](REPORT-v4.md): result under the redefined acceptance. TP2 relies on the existing
  accepted text and finish-reason consistency; TP4 completed token-level verification (every ROW x
  native single-request pair 200/200). PASS.

Checksums: SHA256SUMS-v1 to -v3 describe earlier uploads; SHA256SUMS-v4 covers the current tree.
