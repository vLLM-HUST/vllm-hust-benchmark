# GDN State Codec

GDN State Codec is a bounded Ascend 910B2 state-compression experiment for the
Qwen3.5/Qwen3.8 gated-delta recurrent state. This repository separates the
mechanism, exactness checks, workload evidence, and rejected prototypes from
the StateAxis engine integration.

## Repository metadata

- Owner: `vLLM-HUST`
- Canonical repository: `vLLM-HUST/gdn-state-codec`
- Mod ID: `org.vllm-hust.gdn-state-codec`
- Directly responsible: Shuhao Zhang (Tony) (`ShuhaoZhangTony`)
- Advisor status: confirmed none (`advisor_status: none`, `advisors: []`)
- Hardware scope: Ascend 910B2
- Default state: disabled
- Qualification: unqualified experiment; version 0.2.0 has no retained
  performance workload
- Latest candidate: quant-only vector replace; mixed sub-percent total-time
  deltas, slower TTFT, changed output hashes, and no multi-request support

`MOD_METADATA.json` is the common vLLM-HUST MOD summary. `catalog.json` is the
mechanism-specific machine-readable status record. `PROVENANCE.json` binds
the canonical repository, archived source snapshots, experiment commits, and
evidence digests. This repository is the authoritative home of the mod.

## Status

`unqualified-experiment` — disabled by default.

The current 0.2.0 identity was retested on four Qwen3.5-35B/AgentX first-turn
shapes (328/457/648/1033 prompt tokens, 128 output tokens). Three-run median
total-time deltas ranged from 0.60% faster to 0.81% slower, TTFT regressed in
all four cells, and every output hash differed from native. A true
two-request probe failed closed and terminated the engine. Version 0.2.0 is
therefore not performance-qualified. See
`evidence/qwen35-agentx-v020-expanded-20261009/`.

Historically, the retained v68 path passed clean-build component
correctness/no-harm gates
and two matched Qwen3.5-35B/AgentX single-request eager evaluations. The first
formal bracket measured a 0.92% median end-to-end improvement and 1.09% lower
median ITL; the clean-main retest measured 1.16% and 1.69%, respectively, but
its independent bootstrap intervals overlap zero. These numbers apply only to
the recorded TP2, 649-prompt/128-output first-turn workload. They are not a
general online, graph-mode, multi-request, memory-capacity, or model-family
claim, and it is not inherited by version 0.2.0.

An expanded four-shape AgentX round did not qualify a more aggressive
boundary-refit path. A cheap preserve-compensator variant passed its component
gate and reached step 65 instead of step 22, but then exceeded the unchanged
2% real-shadow RMS limit. Exact SVD passed real-shadow correctness but was far
too expensive. No replace-mode or end-to-end speedup claim was made. See
`evidence/qwen35-agentx-expanded-20261008/` for the negative result and bound
artifact hashes.

A subsequent paper-alignment round implemented an every-window rank-four
power-iteration refit directly from the encoded state on Ascend Cube. On a
frozen Qwen3.5-35B AgentX layer-0 fixture, one power iteration kept boundary
reconstruction RMS at `7.23e-05`; an 82-step production-operator comparison
passed the 2% correctness gate with `0.884%` maximum output RMS. The compressed
steady step took `0.566x` the native time, but five boundary refits made the
matched 82-step total `1.583x` native, so the no-harm gate failed. This path is
not enabled and does not replace the narrowly qualified v68 result above. See
`evidence/paper-alignment-20261008/` for the source comparison, profiler
breakdown, hashes, and decision.

A follow-up rank scan found that one fitted compensator remained inside the
2% exactness gate for the frozen Qwen3.5-35B/AgentX layer-0 fixture. Tracking
the active old rank then removed three unused factor paths after the first
boundary. Three alternating-order 82-step component repetitions measured a
median `0.9767x` compressed/native ratio (about `2.33%` lower elapsed time),
with `0.9449%` maximum output RMS. This is a component candidate only: it is
still disabled and needs longer randomized and real-online validation. See
`evidence/rank1-active-rank-20261008/`.

The next round initialized the fixed rank-four ABI directly at fitted rank one
and tested the zero-iteration fast boundary. On the 82-step TP-local component
gate it measured `0.4380x` native elapsed time (about `56.2%` lower) with
`1.034%` maximum output RMS. It also completed a harder 328-prompt/128-output
AgentX real-shadow request across all Qwen3.5-35B GDN layers with `1.334%`
worst logged RMS. Replace mode did not qualify: five repetitions had a
`30.680 s` median versus `28.750 s` for native (`+6.71%`), despite similar
median ITL. The result is therefore a correct component optimization and an
online negative, not a new end-to-end claim. Quant-only was faster in the
component probe but failed real shadow at layer 18, step 70 (`2.393%`). See
`evidence/rank1-fast-boundary-20261008/`.

The old 576-head boundary-scan/refit prototype failed the no-harm gate by more
than two orders of magnitude. Its patch and negative measurements remain under
`evidence/all-head-negative-m2/`; the exact archived StateAxis commit is also
preserved as an mbox patch under `archive/`. It is evidence, not active code.

## Layout

- `native/ascendc/gdn_state_codec/`: current AscendC kernels, host
  runtime, and component probes from the initial engine-integration snapshot.
- `native/include`, `native/src`, `native/tests`: portable state-capsule and
  Qwen GDN mapping reference with exactness/lifecycle tests.
- `native/specs/`: frozen mechanism and admission contracts.
- `evidence/qwen35-agentx-v68/`: summaries for the matched real-online result
  and clean-main retest.
- `evidence/ecpa-manager03-launch-20261009/`: real Manager 0.3 discovery,
  activation, TP2 worker-effectiveness, request, and shutdown smoke evidence.
- `evidence/qwen35-agentx-v020-expanded-20261009/`: current-identity
  four-shape matched online negative, output-hash comparison, concurrency
  failure, and raw artifact bindings.
- `evidence/qwen35-agentx-expanded-20261008/`: four-shape/concurrency probe,
  boundary-refit negative result, and immutable artifact bindings.
- `evidence/paper-alignment-20261008/`: paper-aligned encoded Cube boundary,
  model-derived scan, matched 82-step gate, and profiler conclusion.
- `evidence/rank1-active-rank-20261008/`: rank scan, active-old-rank
  specialization, repeated alternating-order matched component result.
- `evidence/rank1-fast-boundary-20261008/`: direct rank-one initialization,
  fast-boundary real-shadow pass, matched online negative, and rejected
  quant-only result.
- `evidence/all-head-negative-m2/`: preserved rejected all-head experiment.
- `tools/agentx_online_probe.py`: overwrite-safe streaming AgentX probe that
  rejects truncated HTTP-200 SSE responses.

## Launch with vLLM-HUST Extension Manager

The codec is an independent Manifest 0.3 vLLM plugin. It must be discovered,
configured, enabled, and launched by `vllm-hust-ext`; StateAxis is neither its
launcher nor a runtime dependency. Install the codec wheel and Extension
Manager in the same Python environment as vLLM, then provide an absolute
directory containing the matching native libraries:

```bash
vllm-hust-ext extension inspect org.vllm-hust.gdn-state-codec
vllm-hust-ext extension configure org.vllm-hust.gdn-state-codec \
  --file gdn-state-codec-manager-config.json
vllm-hust-ext extension enable org.vllm-hust.gdn-state-codec
vllm-hust-ext extension check org.vllm-hust.gdn-state-codec
vllm-hust-ext run --dry-run -- vllm serve /path/to/Qwen3.5-35B-A3B \
  --enforce-eager
vllm-hust-ext run -- vllm serve /path/to/Qwen3.5-35B-A3B --enforce-eager
```

`gdn-state-codec-manager-config.json` contains only Manager-owned overrides:

```json
{
  "environment": {
    "GDN_STATE_CODEC_LIBRARY_DIR": "/absolute/path/to/release/libraries",
    "VLLM_HUST_GDN_STATE_CODEC_ENABLE": "1"
  }
}
```

The Manifest supplies the bounded `gdn_state_codec` additional configuration
and defaults to correctness-observing `shadow` mode. ECPA discovery and even
Bundle enablement remain inert until the explicit enable variable is set;
`VLLM_HUST_GDN_STATE_CODEC_KILL_SWITCH=1` always wins. The plugin rejects a
missing or relative library directory, non-NPU devices, non-Qwen3.5 generation
architectures, graph mode, and speculative decoding before native dispatch.

## Validate the portable reference

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The AscendC component is configured separately and requires CANN 9.1.0 on
Ascend 910B2:

```bash
cmake -S native/ascendc/gdn_state_codec -B build-ascend \
  -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DSOC_VERSION=ascend910b2
cmake --build build-ascend --parallel
```

## Provenance and integration

This repository is the canonical mod repository. The initial
engine-integration snapshot is bound to commit
`83707092ecec1eba50d4ea33ea2da615ceaa4fd8`; later experimental source and raw
evidence are bound by commit and SHA-256 in `PROVENANCE.json`. Promoting or
enabling this mod requires a new matched exactness and real-online result; the
repository name alone conveys no broader performance qualification.

Version 0.2.0 replaces the former paper-derived project identity with the
organization-owned `org.vllm-hust.gdn-state-codec` identity and
`gdn_state_codec` API surface. This is an intentional source/ABI migration, not
an alias: active integrations must update their ID, imports, library names, and
configuration. Historical evidence and archived patches remain byte-for-byte
unchanged. See `MIGRATION.md` for the exact mapping.

## Authorship and responsibility

The preserved implementation, experimental iterations, evidence curation,
and this independent mod repository are authored and directly maintained by
Shuhao Zhang (Tony) (`ShuhaoZhangTony`). No advisor role applies. This is
recorded explicitly as `advisors: []`, matching the ownership convention used
by the other mods rather than leaving advisor metadata unknown or inferred.
