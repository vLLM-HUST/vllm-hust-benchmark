# StateAxis MOD ON/OFF protocol

StateAxis MOD performance comparisons use the same fail-closed rules as formal
`vllm-hust-benchmark` campaigns. An importable descriptor is not an ON arm.

Before any device run, each MOD must have an extracted implementation, an
active Manifest 0.3 implementation object, a non-empty activation entry point,
and an activation contract accepted by the Extension Manager. A rejected ON
arm is recorded as `ON_NOT_RUNNABLE`; it is not reported as 0% overhead or 0%
gain, and an OFF-only baseline is not presented as a matched result.

Admitted comparisons require:

1. one frozen parent runtime, backend, model revision, dataset/workload and
   request arrival sequence;
2. three fresh service processes per arm in `OFF/ON`, `ON/OFF`, `OFF/ON`
   order;
3. activation telemetry proving that ON executed the intended mechanism and
   OFF did not;
4. correctness/exactness, lifecycle, resource-release and failure-recovery
   checks;
5. request and output-token throughput, TTFT/TPOT/E2E p50/p95/p99, peak HBM
   and error rate;
6. preservation of failures, regressions and rejected cells.

The workload must exercise the mechanism. A result transfers only to the
declared model, topology, graph mode, concurrency, input/output distribution
and state-reuse pattern. A single blanket workload is not evidence for every
MOD.

Run the admission audit with:

```bash
python scripts/audit_stateaxis_mod_onoff_readiness.py \
  --catalog /root/stateaxis/mods/topic-mod-repositories.json \
  --repos-root /root/stateaxis-topic-mods/repositories \
  --output-dir reports/stateaxis-mod-onoff-preflight-20261010 \
  --npu-snapshot /path/to/npu-smi.txt
```

Exit code `3` means the audit completed but at least one MOD cannot legally
produce an ON arm. It is an expected fail-closed result, not a script failure.
