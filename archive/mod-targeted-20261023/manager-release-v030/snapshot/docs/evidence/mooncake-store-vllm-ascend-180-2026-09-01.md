# Mooncake Store vLLM/Ascend acceptance — 2026-09-01

This acceptance used one isolated container and Ascend NPU 4 on host `180`.
The existing four-rank vLLM service on NPUs 0–3 was not modified or restarted.
The test API bound only to `127.0.0.1:18084`.

## Fixed inputs

- vLLM source base `43341b177dbaa8c7f04662f71e885ee7dfe22704`;
- vLLM compatibility patch `aa2781f7bc` on branch
  `feature/mooncake-store-ascend-kv-cache`;
- vLLM `0.23.0`, vLLM Ascend `0.19.1.post1.dev474+g4edbc9258`;
- `mooncake-transfer-engine-npu==0.3.11.post1`;
- model `Qwen3-0.6B`, one NPU, 2,048-token maximum context;
- local vLLM prefix caching disabled;
- `MooncakeStoreConnector`, `kv_both`, `load_async=true`;
- Mooncake embedded Store with a 1 GiB global segment, 256 MiB local buffer,
  `P2PHANDSHAKE` metadata, and the NPU-aware `ascend` transport.

The NPU wheel did not contain the `xxhash` dependency required by this vLLM
source's default cache hash. The isolated test container installed `xxhash`
explicitly. This is a reproducibility dependency, not an Extension Manager
side effect.

## Core compatibility finding and patch

vLLM Ascend exposes ordinary attention caches as a tuple of separately
allocated K and V tensors. The upstream Mooncake Store worker accepted only a
Tensor or list and asserted before the first request. Taking the first tuple
member would silently omit V and was rejected as unsafe.

Patch `aa2781f7bc` makes the Store worker register every non-null tuple member,
while preserving the existing Mamba-list representative behavior and physical
storage deduplication. The complete Mooncake Store worker test file passed:
75 tests, and Ruff passed for the two changed files.

## Data-path result

Two identical 1,153-token requests completed successfully. Connector metrics
after the second request reported:

- `lookup_exists`, status `ok`: 18 keys;
- `save_put`, status `ok`: 9 keys, 133,191,072 bytes;
- `load_get`, status `ok`: 9 keys, 133,191,072 bytes;
- failed keys: 0.

Because local prefix caching was disabled, the second request's nine-key
`load_get` is a real Mooncake Store hit rather than a local vLLM cache hit.

## Degradation and recovery

The test operator stopped only the isolated Mooncake master. A UUID-distinct
request still returned HTTP 200 from vLLM, but four asynchronous Store saves
became `save_put{status="partial_failure"}`. Therefore vLLM `/health` alone is
not sufficient evidence of a healthy external KV path.

The operator restarted the master without restarting vLLM. A new request saved
four keys successfully, and its repeat loaded the same four keys. The final
counters were 13 successful save keys, 13 successful load keys, and the four
historical outage failures. This proves recovery while also showing that
historical cumulative failures must be evaluated over an observation window.

The Manager did not start, stop, restart, or clear Mooncake. Those lifecycle
actions belonged to the explicit test operator. The isolated container is
removed after evidence and source changes are persisted.
