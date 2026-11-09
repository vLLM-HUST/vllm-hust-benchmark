# PyramidKV standalone 0.4.0rc1 functional release acceptance

Publication date: 2026-11-03 (Asia/Shanghai). Server logs use UTC.

This receipt qualifies the independently packaged candidate on the pinned Qwen3.5-35B-A3B
text/BF16/TP2 Ascend profile. It is not a new task-quality, speedup, HBM-saving or maximum-capacity
result. Historical shared-adapter results retain their own source identities.

## Artifact identity and order

- Source: `bdc20e248c9202560bf44c086bcdede2952b3274`,
  [PyramidKV PR #30](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/pull/30).
- [Public candidate](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/releases/tag/v0.4.0rc1),
  with wheel, sdist, provenance and checksums; no PyPI publication.
- Wheel SHA256: `46a0a9b2dd02b7523fe61d5032a78abd7dceb653b660d0304f2dc52d9ae5ae81`.
- Built the exact merged tree; checked all package files against source and launch/config/docs in
  the sdist. Tested those installed wheel bytes first, then published and anonymously downloaded all
  four assets. Every downloaded byte matched the tested artifact, followed by a fresh clean-venv
  install/discovery check. The serving runs preceded publication; public readback is byte identity,
  not a second serving series.
- Release source CI passed on Python 3.10/3.11/3.12. The prior complete host suite was 206 passed /
  3 skipped. The final installed wheel passed two real NPU oracles; clean disabled discovery imports
  no host/torch/KVCompress dependencies.

## Completed functional lifecycles

Each arm started a fresh server. Only compression activation and the explicit metadata flag vary.
All use the same aligned1368/beta2 configuration, BF16/TP2, 32K, APC, MTP2, async/chunked prefill,
FULL_AND_PIECEWISE capture sizes and equal reserved KV budget. The disabled arm uses the same
explicit host-compatibility worker, with no compression plugin activation or scheduler/worker
compression commits.

| Arm      | Successful requests | Compression transactions with both TP ranks | Repeated-request APC tokens | Smoke MTP draft / accepted delta |
| -------- | ------------------: | ------------------------------------------: | --------------------------: | -------------------------------: |
| disabled |                   7 |                                           0 |                        2048 |                          10 / 10 |
| native   |                   7 |                                           6 |                        2048 |                          14 / 10 |
| batched  |                   7 |                                           6 |                        2048 |                          14 / 10 |

Each completed arm includes short/long/repeated-long requests and four simultaneous unique long
requests. Exact outputs and raw responses are preserved. The two enabled arms have six
request-matched scheduler commits each, with both worker acknowledgements and semantic/physical
length checks. Device-release receipts follow each completed lifecycle. Concurrent HTTP success
establishes only this configured functional operating point.

Host identity records the actual physical 910B2 devices 6/7 (visible 0/1), aarch64/8CPU/64GiB
cgroup, clean pinned vLLM/Ascend source identities and installed package versions. The model
identity manifest is reused from the prepared runtime; this receipt does not claim a new full
weight-file hashing pass. The startup-only model-file cache advice stops at readiness and does not
run during requests.

## Failures and limitations retained

The first disabled attempt completed three correct requests and APC/MTP checks, then the
orchestration script failed before concurrent dispatch because a function shadowed the `concurrent`
module. Its three requests, original driver, failure and shutdown records remain under
`attempts/disabled-controller-error`; they are additional to the 21 requests in the completed table.
The driver binding was corrected, and all three final lifecycles used that same corrected driver. No
plugin or host code was changed for this script error.

An earlier bare `pytest` command selected the system interpreter and failed collection because the
installed overlay was absent. Both that error and the corrected installed-wheel `python -m pytest`
NPU result are retained. A preceding environment interruption erased an unfinished temporary
preflight; no result is claimed for it. These preparation failures are not task-quality
measurements.

The batched arm selects the opt-in path; the NPU oracle verifies its native slot equivalence.
Serving success alone is not a profile of every internal branch or a speedup measurement. No new
LongBench, five-dataset score or Agent execution is claimed. Text-only serving does not qualify HLE
image-bearing subsets.

## Offline verification

Run from this directory:

```bash
sha256sum -c SHA256SUMS
python verify.py
```

The verifier checks every raw member's size/hash, request identities/outputs and equal payloads, all
enabled scheduler/TP acknowledgements, disabled nonactivation, APC/MTP deltas and separate
processes. It rejects optimized Python. Original logs and timing values are retained for audit, not
promoted to a benchmark effect. The retained failed preliminary attempt is separately named and
excluded from the completed-lifecycle table.

Current-version effect work is tracked in
[PyramidKV #31](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/31); resource and
calibrated-judge handoffs in
[#32](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/32) and
[#33](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/33). Parent #1/#8 remain open.
The canonical plan is [here](../../docs/plans/pyramidkv-standalone-qualification-20261103.md).
