# ECPA matched process gate — 2026-10-10

This campaign runs three interleaved OFF/ON pairs with a fresh StateAxis
`native_state_engine` and Native protocol worker for every arm. Extension
Manager owns activation and injects the Manifest 0.3 `--additional-config`.

The workload sends eight identical unretained cold prompts with a stable cache
salt. ON must record one cold insertion followed by seven exact hits; OFF must
remain behaviorally and observably disabled. Output tokens, cache ownership,
resource release, MOD identity, collision/stale counters, and process shutdown
are checked on every arm.

This is deterministic host/process evidence only. The protocol worker does not
execute on an accelerator, so timing fields support no latency, throughput, or
NPU claim.

Run from a clean checkout after building `native_state_engine --release`:

```bash
python evidence/ecpa-matched-process-20261010/run_matched_campaign.py
```
