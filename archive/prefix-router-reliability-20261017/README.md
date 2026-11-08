# Prefix Router reliability evidence capsule

This capsule preserves the measured and artifact-specific statements removed
from `vLLM-HUST/vllm-hust-prefix-router#5` under the organization's centralized
benchmark custody policy. It is a historical, non-formal record: the referenced
native archives are owner-held and were not published with the PR, so their
contents cannot be independently replayed from this repository.

## Publicly reproducible CPU/package gates

| Gate | Recorded result |
| --- | --- |
| Standalone recovery increment `00e5e76` | 94 CPU tests passed on Windows / Python 3.14.5 |
| Complete source `ab8dcc5` | 270 passed and six Linux-only skips; unpacked sdist matched |
| Installed wheel outside checkout | 236 passed and four skips |
| Linux / Python 3.12 source and wheel | 276 and 238 passed, with two source-only skips |
| Earlier Linux package on Python 3.10/3.12/3.14 | 276 source and 238 wheel passes per version |
| Deterministic HTTP fault matrix | 48 cells and 576 requests using synthetic backends, virtual time and sequential arrivals |

The rebased current candidate was independently rerun on 2026-11-08: Ruff and
format checks passed; 279 tests passed and one was skipped in an isolated Python
3.12 environment. These are CPU/package checks, not NPU acceptance.

## Native normal-path background

The 2026-10-15 follow-up recorded source
`ab8dcc51c127abb74afd54cb533081be17331237` and wheel SHA-256
`2b016751957bb8ffab748a92b72b03f72466e92d6705d96bcdff34924b1dcf3a`.
The setup was Qwen2.5-14B-Instruct, FP16/eager, two TP1 backends, Python 3.12.13,
8 CPU bandwidth and 64 GiB. The 188-package freeze digest was
`ec45b6f95b04db4c677d7b4449bf389572666fba0a3a12303746bd0227c38015`.

Four fresh backend blocks recorded 384 measured, 384 warmup and 32 seed
requests. Router/direct block or cell median E2E deltas ranged from -4.27% to
+6.45%; TTFT deltas ranged from -0.14% to +9.19%. The predefined repeated 10%
E2E warning did not fire. Blocks, not requests, were the independent units.
The raw-cell digest was
`2988426762298761417c0f08a5bb67849a5dc5f7cd1b54594b51a75154c56680`.

These observations do not establish speedup, isolated reliability overhead,
P99/throughput, capacity, real multi-DP behavior or a supported host range.

## Earlier fault background and retained failure

The earlier runs used wheel SHA-256
`d4f50f6bdb081ac34176bcdd31ea807ac37d690c6db407c1ee6f58e956df5e58`.
A four-pattern/two-mode repeat offered 192 requests: 190 completed and two
bounded in-flight failures. Eight recovery probes observed cold-0 to cached-384
transitions. Restart-to-view-ready was 126.77–147.23 seconds including reload.

The first restart attempt failed its oracle. One survivor replay timed out near
three seconds while adjacent replies were near 0.1 seconds. Its 48 requests and
two probes completed, but the cell remains failed. Raw-cell SHA-256:
`3c851b4966e6a0befc3d06d12719d8921d9f11c82cb573c780828c8d3a8600a1`.

A later diagnostic completed 48 requests, two probes and 814 replay
transactions. About 42% of sampled CPU scheduling periods were throttled, with
loop-delay maxima of 194.83/147.31 ms. This did not isolate the cause or justify
changing timeouts. The latest wheel did not rerun every fault.

## Owner-held archive identities

| Archive | SHA-256 |
| --- | --- |
| `20261013-pr4-integration-v1-reviewed.tar.gz` | `6564698e13e1e0142ec5dedb924aa6144dc4a6b0c5be834e2fbca8c46abcf6a2` |
| `20261013-pr4-replay-review-delivery.tar.gz` | `deded5b3f7e72620510c6de210e2a9bf42d9197865192bb976e60a2d279e0939` |
| `20261015-pr4-final-wheel-reviewed.tar.gz` | `689d907113d3acee283aca6f3daa11992b0be6e49315e4870a2c39ce76d6b270` |

No archive bytes are present here. Do not treat these hashes as download links
or promote the observations to formal qualification without reviewer-accessible
raw data and an independent replay.
