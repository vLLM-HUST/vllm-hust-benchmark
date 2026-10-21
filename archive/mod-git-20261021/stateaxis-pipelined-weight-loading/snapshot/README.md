# stateaxis-pipelined-weight-loading

Extension ID: `org.vllm-hust.stateaxis-pipelined-weight-loading`

Experimental, default-off authority for the StateAxis Qwen2 startup loader.
Extension Manager/ECPA verifies the research digest. StateAxis then admits only
a canonical `pinned_double_buffer_async_v1` plan whose SHA-256, packed-weight
root, geometry and execution-artifact-v6 identity all agree. The validated
absolute plan path is injected only into the child worker; without ECPA
authority the host removes the ambient variable and the worker retains its
`serial_pageable_sync` default.

The candidate uses two 64 MiB pinned staging slots, incremental full-byte
SHA-256, asynchronous H2D, event-guarded slot reuse, per-pack verification and
transactional Ready. Any plan, artifact, root, geometry or worker-start failure
fails closed.

## Evidence boundary

The frozen Qwen2.5-14B real-NPU 3+3 cells measured process-to-Ready reductions
of 84.340% cold and 88.796% warm. TPOT P50 changed +0.043% cold and -0.685%
warm, within the pre-registered 1% regression gate. A separate MiniMax-M2.5 EP8
loader 1+1 cell had a 54.748% process-time point estimate, but was not full-model
serving. These are workload-scoped results, not general qualification.

Current ECPA evidence validates the activation boundary, exact identity chain,
default-off environment stripping, ON-only worker injection, counters and
failure closure. See `EVIDENCE.md`.

A newer Qwen3.5-35B-A3B BF16 TP2 carrier diagnostic tested rank-local
pre-sharded checkpoints on two Ascend 910B2 NPUs. Across three alternating
same-host/same-filesystem repetitions, median process-to-Ready fell from
337.70s to 301.62s (10.68%) and median weight iteration fell from 166.20s to
107.15s (35.53%), with identical first-output hashes. This is promising
workload-scoped carrier evidence only: it was not activated through ECPA, cache
state was uncontrolled, and it makes no AgentX steady-serving claim.

```bash
python -m pip install -e '.[test]'
pytest -q
VLLM_HUST_EXT_CONFIG=evidence/ecpa-contract-20261011/manager-config.json \
  vllm-hust-ext extension check org.vllm-hust.stateaxis-pipelined-weight-loading
```

Maintainer: Shuhao Zhang (Tony), directly responsible; no advisor is declared.
