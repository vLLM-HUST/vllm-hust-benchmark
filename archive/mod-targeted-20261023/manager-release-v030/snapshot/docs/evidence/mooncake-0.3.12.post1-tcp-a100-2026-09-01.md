# Mooncake 0.3.12.post1 TCP acceptance on A100

Date: 2026-09-01 (Asia/Shanghai)  
Host: `a100-dev`  
Distribution: official `mooncake-transfer-engine-non-cuda==0.3.12.post1`  
Wheel SHA-256: `691b4df2a74e32fd9b1877317097d26fd8c5f48692fba920caf5e3a518f36911`

This acceptance was intentionally CPU/DRAM/TCP-only. The one-shot containers
were started without GPU device arguments and emitted the expected warning that
no NVIDIA driver was visible. The host's unrelated user vLLM workload occupied
72,091 MiB on each GPU before and after the final probe. No Mooncake process,
container, Store object, or shared-memory allocation was left running.

## TransferEngine path

`tests/e2e/mooncake_tcp_p2p_probe.py` ran two independent official
`TransferEngine` processes with `P2PHANDSHAKE` metadata and TCP transport. The
sender registered a 1 MiB DRAM buffer filled with byte value `1`, issued
`transfer_sync_write`, and the receiver verified every byte plus the first and
last byte. The transfer returned `0`; both processes unregistered memory and
removed their segments.

This proves a real TransferEngine DRAM/TCP data path. It does not prove
Mooncake Store object semantics or a vLLM connector hit.

## Mooncake Store object path

`tests/e2e/mooncake_store_rest_probe.py` launched an isolated official
`mooncake_master`, the master's embedded HTTP metadata server, and
`mc_store_rest_server` on loopback-only test ports. The Store contributed a
64 MiB segment and used a 16 MiB local buffer. A UUID-scoped key containing a
57-byte value completed:

| Operation | Result |
| --- | --- |
| `PUT /api/put` | HTTP 200 |
| `GET /api/exist/<key>` | HTTP 200, `exists=true` |
| `GET /api/get/<key>` | HTTP 200, byte-for-byte match |
| `DELETE /api/remove/<key>` | HTTP 200 after the configured lease expired |
| `GET /api/get/<key>` after remove | HTTP 404 |

The first diagnostic run tried ordinary remove immediately after get and
correctly received HTTP 500 because a read renews Mooncake's hard object lease.
The final probe configured a 1,000 ms lease, waited 1.25 seconds, and then
removed only its own random key. It never called `remove_all` or force deletion.
This is a concrete reason the Manager must not treat data deletion as an
adapter lifecycle operation.

## Provider and schema consequences

- Mooncake remains owned by `external_operator`; the Manager only renders the
  official vLLM connector configuration and checks operator-supplied health.
- The range at the time of this standalone probe was
  `>=0.3.12.post1,<0.4`; subsequent official NPU `0.3.11.post1` evidence
  widened the experimental profile to `>=0.3.11.post1,<0.4`.
- Mooncake Store REST and vLLM's KV transfer configuration do not publish
  independent protocol semantic versions. Their manifest ranges are therefore
  explicitly `null` instead of invented `1.0` values; package/host ranges and
  executable acceptance evidence govern compatibility.
- `MooncakeConnector` and `MooncakeStoreConnector` remain host-built-in
  carriers. No unofficial `vllm.*` entry point or Mooncake C++ plugin ABI was
  added.

## Remaining release gate

This standalone probe did not observe a vLLM connector hit. That separate gate
was subsequently completed on Ascend and is recorded in
`mooncake-store-vllm-ascend-180-2026-09-01.md`. At the time, publication
remained frozen for other host contracts. The later v0.3.0 decision is recorded
separately in `../release-readiness.json`; it does not broaden this Mooncake
result beyond the pinned matrix above.
