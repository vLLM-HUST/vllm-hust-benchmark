# stateaxis-automatic-prefix-cache

Extension ID: `org.vllm-hust.stateaxis-automatic-prefix-cache`

Automatic content-addressed prefix reuse with lease-safe bounded LRU residency.

This repository is the independent MOD boundary for StateAxis issues [#55](https://github.com/Qixin-Gaoke/stateaxis/issues/55), [#95](https://github.com/Qixin-Gaoke/stateaxis/issues/95).
It is default-off and is admitted only through the vLLM-HUST Extension Manager
StateAxis provider in explicit experiment mode. The split does not inherit
correctness, device, performance, or publication qualification from the aggregate
StateAxis repository.

## Evidence boundary

Status: **historical no-harm in one cell; independent implementation unqualified**.

The frozen 3+3 closure was token-exact and lifecycle-clean with 0.0256%
throughput overhead. Its candidate arms recorded eight insertions and six evictions
but zero hits, so it establishes bounded ownership safety rather than a speedup.
The independent implementation therefore remains performance-unqualified until a
new matched repeated-prefix campaign exercises authoritative hits.

The copied evidence and its SHA-256 are recorded in `PROVENANCE.json`. Negative,
failed, and inconclusive results are retained. Microbenchmarks and component results
must not be restated as online end-to-end gains.

## Install and inspect

```bash
python -m pip install .
vllm-hust-ext extension inspect org.vllm-hust.stateaxis-automatic-prefix-cache
vllm-hust-ext extension check org.vllm-hust.stateaxis-automatic-prefix-cache
```

Discovery alone does not enable the MOD. The active Manifest 0.3 contract is
accepted only with `experiment_mode=true`, the exact `RESEARCH_MANIFEST.json`
digest, and the matching StateAxis host identity. The independently reviewable
content-addressed, collision-verified, lease-safe LRU index lives under `native/`.
StateAxis remains responsible for resident-state validation, device eviction,
lifecycle cleanup, and fail-closed rollback.

## Validate

```bash
python -m pip install -e '.[test]'
pytest -q
cargo test --manifest-path native/Cargo.toml
```

Maintainer: Shuhao Zhang (Tony), directly responsible; no advisor is declared.
