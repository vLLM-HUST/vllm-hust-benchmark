# stateaxis-async-kv-transfer

Default-off StateAxis MOD for a bounded asynchronous host-to-device KV restore
lane. The MOD supplies an active Manifest 0.3 contract and a reusable Rust
coordinator; StateAxis owns the device stream, snapshot memory, and publication.

The coordinator admits only nonzero `(state_id, generation, snapshot_sha256)`
identities, coalesces exact duplicates, bounds both unique work and correlation
IDs, issues monotonically changing leases, and requires explicit commit or
abort. The StateAxis activation contract sets the retry budget to zero: any
device-transfer error rolls back and fails closed rather than replaying DMA.

The mechanism is not enabled by importing this package or setting an environment
variable directly. It must be selected by vLLM-HUST Extension Manager/ECPA and
accepted by StateAxis's hash-bound `--additional-config` parser.

Extension ID: `org.vllm-hust.stateaxis-async-kv-transfer`

## Evidence boundary

Status: **historical matched real-NPU component-positive; active version
performance-unqualified**.

The frozen StateAxis R5 campaign ran Qwen2.5-14B BF16 on one Ascend 910B2 in
three fresh processes per arm. All six runs passed exactness, lifecycle and state
identity checks. Candidate host-tier-hit p50 fell from 592.597 ms to 333.716 ms
(-43.686%); concurrent-pair makespan p50 fell from 607.648 ms to 355.473 ms;
pair throughput rose from 3.291 to 5.626 ops/s. Candidate device overlap was
observed in 3/3 runs and peak HBM remained within the same 44,174--44,177 MiB
range.

This is a directional, workload-scoped result for a frozen historical
implementation. It does not establish statistical significance, general serving
throughput, longer-context behavior, capacity scaling or qualification of MOD
version 0.2.0. The current revision has separate 3+3 matched-process exactness and
lifecycle evidence, but that deterministic worker does not execute accelerator
kernels. See `EVIDENCE.md` and the machine-readable R5 bundle under
`evidence/issue76-async-transfer-20260815-r5/`.

## Validation

```bash
python -m pytest -q
ruff check .
ruff format --check .
cargo test --manifest-path native/Cargo.toml
cargo clippy --manifest-path native/Cargo.toml --all-targets -- -D warnings
```

See `EVIDENCE.md` for the strict boundary between historical evidence and fresh
version qualification.

Maintainer: Shuhao Zhang (Tony), directly responsible; no advisor is declared.
