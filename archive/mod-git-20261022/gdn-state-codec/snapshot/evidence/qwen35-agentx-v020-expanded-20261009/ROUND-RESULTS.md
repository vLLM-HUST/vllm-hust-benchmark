# GDN State Codec 0.2.0 expanded AgentX retest

Date: 2026-10-09
Evidence labels: `matched-real-online-negative`, `exact-output-mismatch`,
`concurrency-fail-closed`

## Result

The current `org.vllm-hust.gdn-state-codec@0.2.0` replace path does not qualify
on this workload. Four Qwen3.5-35B-A3B AgentX first-turn shapes were measured
against an unchanged native baseline with one warmup and three retained runs
per cell. Candidate total-time medians ranged from 0.60% faster to 0.81%
slower, while TTFT regressed in all four cells by 2.09% to 4.43%. Those mixed
sub-percent total-time changes are not a repeatable end-to-end improvement.

The candidate's 128-token output hash differed from native in all four cells.
This comparison does not include a task-level quality score, so it records an
exact-output mismatch rather than claiming semantic failure or equivalence.

| Prompt tokens | Native total | Candidate total | Candidate delta | TTFT delta | ITL delta | Exact output |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 328 | 27.7963 s | 28.0203 s | +0.81% | +3.42% | +0.37% | no |
| 457 | 27.9188 s | 28.0400 s | +0.43% | +2.09% | -0.27% | no |
| 648 | 28.1440 s | 27.9760 s | -0.60% | +3.72% | -0.65% | no |
| 1033 | 28.0984 s | 28.0837 s | -0.05% | +4.43% | -0.47% | no |

Positive deltas are regressions. With only three retained repetitions per
cell and effects below 1%, these measurements are descriptive, not evidence
of a stable performance effect.

## Concurrency and graph boundaries

An actual two-request probe (`concurrency=2`, `repetitions=2`) triggered the
runtime gate `GDN State Codec runtime gate requires one decode request`.
Both HTTP 200 streams carried embedded 500 errors and the engine terminated.
This is a preserved negative lifecycle result. A preceding
`concurrency=2`, `repetitions=1` control emitted only one request and is not a
concurrency result.

Graph mode remains inadmissible: the StateAxis parser rejects the mod when
`enforce_eager` is false. The narrow parser suite, including that case,
passed 17 tests. No graph-mode performance claim is made.

## Reproducibility boundary

- StateAxis parent: `588decd53002a3da9f39f79656898bf4fea463ee`
- measured mod source/pin: `c7faafe98298ef620f47fcf03bf1a7b30f36bccb`
- post-measurement probe-only fix: `a36cf068005e5db7f145dd7a4e3902663e53b318`
- model revision: `59d61f3ce65a6d9863b86d2e96597125219dc754`
- model config SHA-256:
  `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`
- hardware: two Ascend 910B2 devices, tensor parallel 2
- runtime: vLLM 0.23.0 StateAxis integration, CANN 9.1.0, eager,
  max model length 2048, GPU memory utilization 0.9
- candidate: `quant_only`, vector decode, replace mode, zero boundary power
  iterations
- binding manifest: SHA-256 of `MOD_METADATA.json` at the measured commit,
  `e08b653672f33fe58eed0aaa68fc015f1de24deafc04d451110f6f5fa878791c`
- custom MoE OPP SHA-256:
  `b77ec1ec0c5e8b8c31f29dd6763979a7f9442a288c7812988655d273a41e2855`
- local raw evidence:
  `/root/stateaxis-evidence/gdn-state-codec/2026-10-09-qwen35-agentx-v020-expanded/`

Both measured repositories were clean. The probe-only fix changes host CLI
argument parsing and does not change the measured runtime or kernels.

## Artifact bindings

- native measurements:
  `1e586fae611b414b38755ab2c42dccc3f2be62975e5b5561619b80e931a5b5fb`
- candidate measurements:
  `a362c0d21b2def3e0560d72166493dc15a1021919b307abe0c240408bdd90d08`
- true concurrency-two negative:
  `558e52e38a174b54a772a00dbc7731a990400fa8a599ab56b78e60db5d568d3d`
- single-request concurrency control:
  `11f7441be4c0e89d8336a59784e015658848e74481ca941966c0ea870a094bb7`
- successful native server log:
  `5cef7996882c0f48693475ef0842601bfd4e3e796656766acda767d54c113282`
- candidate server and concurrency-failure log:
  `45bd8c9b67ac87f1b091172b0d6b967139d4f3215be158ac296fd717370e51eb`
- corrected six-cell component probe:
  `79a5c50fe2602ad53e7f0d11f02e4e9a4ebe14a29b56adfab6be26751c0c4e9b`

Two earlier native startup attempts are retained as setup failures. They used
custom OPP artifacts that lacked `aclnnMoeInitRoutingCustom`; neither produced
performance samples.

## Validation and decision

- AgentX streaming probe tests: 2 passed.
- StateAxis GDN admission/parser tests: 17 passed, 14 upstream warnings.
- clean AscendC Release rebuild: passed.
- corrected component matrix: batch 1/2 by position 0/7/15, all six gates
  passed; positions advanced to 1/8/16 as expected.

Keep 0.2.0 disabled and unqualified. Preserve the historical v68 result as a
historical, commit-bounded observation only; it is not inherited by the new
identity or current code. Before another online timing round, require exact or
task-scored quality, graceful multi-request fallback, and a mechanism whose
component saving is large enough to exceed the observed sub-percent noise.
