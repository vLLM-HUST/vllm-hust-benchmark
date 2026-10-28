# Publication-status correction, 2026-10-28

The `Release upload/public readback remain pending` statements at the beginning and end of
[REPORT-v2.md](REPORT-v2.md), and the previous README wording, were not updated when the evidence
publication completed. This was an editorial error. It does not mean the archived runs or their
quality gates changed.

The original records already committed with that report show:

- [Anonymous download verification](public-download-verification-v1.json): all three assets returned
  HTTP 200 and matched the frozen local SHA256 and size.
- [Archive-member verification](public-member-verification-v1.json): all 212 original members
  verified without extraction or execution.
- [Manifest](MANIFEST-v1.json): the archive is 1,519,854 bytes with SHA256
  `8db06d57771d3c6d0e809d3e139e475ed21f421345fd954ea2ad077ea34ae01e`.

The
[Release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-muxpd-scheduling-20261026)
contains `muxpd-scheduling-20261026-evidence-v1.tar.gz`, `MANIFEST-v1.json`, and `SHA256SUMS-v1`.
The receipts record the verification time on 2026-10-26; this correction is not a new hardware run
or a new current-time download claim.

REPORT-v2, raw inputs/results, manifests, and verification receipts are retained byte-for-byte. This
correction supersedes only the stale publication-status sentences. The async candidate's TTFT/P99
tradeoff, lack of HTTP Goodput/Manager qualification, and unresolved answer-quality Issue #23 remain
unchanged.
