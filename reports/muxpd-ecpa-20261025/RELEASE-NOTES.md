# MuxPD ECPA installed-wheel qualification — restricted TP4 evidence

This evidence release preserves the failed first startup, two completed installed-wheel TP4
functional qualifications, the separate 72-request text-repeat study, package identity checks, and
CPU/CI receipts. It does not publish a leaderboard result or a new performance gain.

The model/host scope is Qwen3.5-35B-A3B BF16, TP4+EP on four Ascend 910B2 devices, with explicitly
pinned HUST Core and Ascend source revisions. Functional checks cover real three-component output
receipts, full-prompt prefill progression, cancellation, process-bound evidence revocation, and a
fresh native launch after disable. Shutdown warnings are retained.

Numerical equivalence is not established. The two qualification rounds matched 7/8 and 6/8 natural
texts. A separate native-then-candidate study completed 72 requests and matched 23/32 concurrent
text pairs plus 2/4 isolated pairs. Both endpoints also varied internally; native variation does not
explain away candidate effects. HTTP/nonempty checks are not a semantic oracle, and no output token
IDs or logits were collected. See
[Issue #23](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/issues/23).

The plugin wheel SHA256 is `7cc6b1d9fc8a2afa90526495a7f33e4d08901001d7c7195fbac3bf9ad06ecd7f`. The
final Manager wheel SHA256 is `b78dd61775e2959adf111e0b531a80e3d5e1bbc10c9df62f5f0e644a640b436e`.
The latter was not directly relabelled as device tested: all 42 runtime package files are
byte-identical to the hardware-tested Manager wheel
`ac0fb620a862314cd6f8510e1d498f61dceff39de9eb94bfba8bc330f8030781`; distribution metadata keeps its
separate identity.

Assets consist of the immutable raw archive, a member index, local archive verification, and
`SHA256SUMS`. Verify the asset digest first, then every decompressed member with the committed
`archive_qualification.py --verify ARCHIVE --index INDEX`. Extracting or verifying evidence does not
execute clients, install packages, or run NPU workloads.

Final Manager remote CI passed on Python 3.10 and 3.12: each recorded 2483 passed, 36 skipped and 6
subtests passed. The local final-source full run recorded 1 failed, 2482 passed, 36 skipped and 6
subtests passed; its unchanged H3 500 ms dispatch gate failed. A targeted pass used the same H3 test
source and threshold on source 894 in an earlier window, while another pytest CPU process was still
running. It was not an isolated causal experiment and does not erase the full-suite failure. Manager
dev release policy remains no-go; this is restricted MuxPD functional qualification, not a general
production release approval. Historical MuxPD Goodput evidence uses a different software identity
and is not promoted to a result for this release.

The separately published
[software prerelease v0.2.0a1](https://github.com/vLLM-HUST/vllm-ascend-MuxPD-hust/releases/tag/v0.2.0a1)
contains both wheels and sdists. Its six assets were downloaded back and verified byte for byte; the
receipts are included here.
