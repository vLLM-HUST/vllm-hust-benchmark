# Qwen3.5-35B TP-sharded loader carrier diagnostic

Evidence label: `matched-real-npu-carrier-diagnostic-not-mod-performance`.

This cell asks whether pre-sharding a Qwen3.5-35B-A3B BF16 checkpoint for the
two tensor-parallel ranks is worth promoting into the ECPA-controlled weight
loading mod. It does **not** claim that the mod was ON: all six launches used
the same unmodified vLLM serving entry point, and only the checkpoint layout and
vLLM load format differed.

## Matched result

| Metric | control (3 runs) | TP-sharded (3 runs) | median reduction |
| --- | ---: | ---: | ---: |
| process to Ready | 420.62, 337.70, 331.10 s | 301.62, 270.75, 302.73 s | 10.68% |
| weight iterator | 224.80, 166.20, 164.53 s | 107.15, 95.70, 126.32 s | 35.53% |
| first request | 4.03, 4.21, 4.30 s | 4.11, 4.17, 4.12 s | 2.11% |

All first responses had SHA-256
`a4bf318f01f7b329d2d8ce12d19ff432d7527c077e3137c5f29f82e0be8b6deb`.
The control and candidate used the same model configuration and index hashes,
the same host, the same overlay filesystem, two Ascend 910B2 NPUs, vLLM
`0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, and vLLM-Ascend
`1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`.

The machine-readable source of truth is `RESULT.json`. Recompute it with:

```bash
python tools/summarize_qwen35_loader.py \
  evidence/qwen35-loader-sharded-qualification-20261012 \
  --output evidence/qwen35-loader-sharded-qualification-20261012/RESULT.json
```

## Boundary and next gate

OS page-cache state was neither reset nor measured, runtime dirty state was not
captured at every launch, and the first candidate run preceded the first
overlay control. Therefore this result is promising carrier evidence, not a
formal ECPA on/off qualification and not an AgentX throughput result.

The next gate is an identical-command, Manifest 0.3/ECPA-controlled 3+3 launch
with activation telemetry, runtime dirty-state capture, output exactness,
failure closure, resource release, and AgentX steady-serving regression gates.
