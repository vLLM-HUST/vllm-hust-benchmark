# LatchMoE five-dataset server audit — 2026-10-10

**No five-dataset LatchMoE quality result or optimization claim is published here.** This report
delivers a reviewable applicability/blocker matrix, real server/model/configuration diagnostics, a
narrow MOD parsing repair and an offline paired-evidence/scorer connector. The matrix is awaiting
maintainer review. LatchMoE #94 remains open.

## What was actually checked

- Connected to user-authorized Coder workspace `VLLM-HUST-LCW`; the initial pod exposed only
  physical NPU5 (Ascend910B2, 64GiB). After a workspace stop/restart it exposed only NPU2, still one
  card. Yesterday's four-card/native-TP2 records are not today's resource inventory.
- Preserved base Python packages, `/root/latchmoe-env`, locked core/seam checkouts and prior
  model/validation directories. Candidate source testing uses new task directories and
  subprocess-local PYTHONPATH. No model service was started, original checkpoint edited, installed
  MOD replaced, external sandbox created, website deployed or PR merged.
- Found the target BF16 model at `/root/qwen35-discovery-20261009/model`. Independently SHA-256
  checked all 24 locked files, including all 14 weight shards, after restart against revision
  `b3a3339bdfee46ccfa27a6c576d12c8eacabfe85`. The earlier interrupted nine-shard scan is preserved
  separately, not used as proof of the complete scan.
- Reproduced the **real installed MOD's config-planning failure** against that actual config:
  `ValueError ... requires a MoE model config with num_hidden_layers`. The loader returned the
  composite wrapper rather than language `text_config`. Full traceback is retained.
- Added regression tests before the fix: server RED = **7 failed / 1 passed**, selected new cases.
  The narrow candidate selects language dimensions/dtype, retains wrapper quantization metadata and
  rejects malformed text configs. Actual config planning then passes; this is **not NPU model
  inference or TP2 qualification**. All existing TP/indexed guards remain.
- Candidate regression logs are separated from real outputs. CPU scorer/contract fixtures are never
  entered into dataset-validation or leaderboard data. See final logs for final counts and source
  hashes; intermediate `03-*` logs preserve the prior formatting-only candidate.

Publication-only checks then moved public revision allowlist comments onto the literal lines. The
exact final gate bytes were retested: `05-benchmark-final.log` passes 152 tests and
`05-postflight.log` records final source hashes and unchanged protected environments. All configured
pre-commit hooks passed on the changed-file scope; this is not full-repository CI.

## Five primary datasets

The exact source identities and local blockers are in `applicability.json`; FrontierScience is split
into two rows without collapsing its tracks. All five are currently **blocked** for this
server/program, not measured negative results and not not-exercised model runs.

| Dataset            | Current adapter boundary                                                         | Additional blocker beyond shared model/runtime/assets gates                                        |
| ------------------ | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| MMLU-Pro           | Strict canonical scorer reuse, Latch-specific removal of compression eligibility | Full/named subset execution contract, actual outputs and live telemetry                            |
| HLE-Verified       | Source and modality audit only                                                   | Scope, rights, images, pinned calibrated judge; Gold has93 image questions                         |
| SWE-bench-Pro      | Frozen Pro V2 source/harness audit                                               | Task rights, AMD64 isolated backend, images, agent/tool budget and fresh regrade                   |
| FrontierScience    | Candidate per-track input/solver/judge primitives                                | Real judge identity, budget and calibration; keep all60 Research source rows                       |
| Terminal-Bench 2.1 | Source audit and Harbor/Terminus configuration candidate                         | Task rights, AMD64 backend/images/resources, actual task verifier and MOD-specific service binding |

The common blockers are:

1. The mandatory cell is Qwen3.5 BF16 **two910B2 / TP2**; this container only has one card.
1. Current LatchMoE capability/seam rejects TP2; indexed eager/graph additionally requires TP1. A
   static descriptor's unresolved router fields do not mean native model construction would reject
   the architecture. TP2 rejection is a separate explicit guard. Shared-expert and GDN code exists
   but their Qwen3.5 MOD cell has not been qualified.
1. The canonical source bundle is asset-frozen elsewhere but absent at the standard path in this
   workspace. Yesterday's small discovery inputs are not the complete reviewed five-dataset
   execution assets. Raw datasets are not redistributed in this PR.
1. No reviewed full execution/scorer/scaffold/tool contracts, live MOD request-window collector or
   final signed release/OCI identity is supplied. HLE/Frontier need approved judge/budget;
   SWE/Terminal need an isolated AMD64-capable backend. Docker/Podman/Harbor and sockets are absent
   here. Installing a binary alone does not provide that isolation/backend.

Owners and unblock conditions: Li-changwu coordinates runtime/resource/assets; pluviophile-chen
coordinates validation; project/canonical evaluator maintainers approve rights, judge and budget;
platform administration provides the required devices and isolated sandbox backend. The last two
roles require human provision/approval, not inferred authority to purchase resources.

## Model/runtime identities and capacity boundary

Model config hash is `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`. The target
has40 language layers,256 routed experts,top8,hidden2048,expert intermediate512, and a gated shared
expert. The14 shard bytes total71,903,878,016; exact tensor payload is 71,903,655,008 bytes. The
recorded tensor-prefix ledger separates language/head, vision and MTP: do not treat checkpoint bytes
as measured process HBM. For ordinary text inference, language+head raw tensor payload alone
is69,321,230,976 bytes, about64.56GiB, before runtime buffers. The fixed official topology
independently requires TP2; no TP1 native OOM is forced just to rediscover the resource gate, and no
baseline-only CPU offload/quantization substitute is used.

Locked hosts: core `762f85b311fbab0bcf8921dd216f5093cd58b9b8`, seam
`2c8c722107a54127999a64c4eb0ec86139df8c26`. Installed MOD wheel from the prior task remains SHA-256
`ebc57f2a29fc09bf5c0e19709925fab0774ec877e1b459bac9f4fc2504401934`. Baseline source review used
benchmark `b32fcbdaf8d3237936b35f9f4ba441bf2d64b361` and MOD
`f2cc19ca6d48a94d2da88378f7c1e2621c6f58b9`. The final MOD parsing candidate is distinct from that
installed wheel and is not promoted into its older Qwen3 qualification.

## Reproduction and evidence

Final verification: **70 MOD-related tests and 152 benchmark/scorer/audit tests passed** on the
server; 54 related MMLU/paired-gate tests passed locally. These are logic/regression tests, not
Qwen3.5 benchmark scores. Final source-only MOD candidate commit is
`ec0c21b76a3f12c64c659146b44efe0585b6d657`. Windows-to-Git newline normalization was checked before
final Linux testing, so final committed Python blobs match the tested bytes.

Server task root: `/root/latchmoe-five-datasets-20261010`. Commands/scripts and final source hashes
are preserved here. `00-server-preflight.log` precedes restart; `05-postflight.log` is the final
inventory and is authoritative for the last observed device allocation. `model-audit.json` records
full independent file hashes and a static capability reproduction. Its checkpoint descriptor
intentionally leaves router ownership unresolved until native layer construction.

For source-only tests use the already provisioned isolated Python; no installer is needed:

```bash
source /root/latchmoe-validation-20261008/input/activate-latchmoe.sh
cd /root/latchmoe-five-datasets-20261010/mod-source
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD" python -m pytest \
  tests/test_autoconfig.py tests/test_configuration_integration.py \
  tests/test_capabilities.py tests/test_model_registry_v2.py -q
cd /root/latchmoe-five-datasets-20261010/benchmark-source
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD/src" python -m pytest \
  tests/test_latchmoe_dataset_pair.py tests/test_mmlu_pro_pyramidkv.py \
  tests/test_frontierscience_pyramidkv.py tests/test_pyramidkv_hle_audit.py \
  tests/test_pyramidkv_swe_audit.py tests/test_pyramidkv_terminal_audit.py \
  tests/test_pyramidkv_terminal_plan.py -q
```

`SHA256SUMS` covers every delivered file except itself. These checksums validate bytes, not human
review, live instrumentation truth, NPU correctness or task-quality scoring. Consult
`docs/LATCHMOE-FIVE-DATASETS.md` for the candidate paired receipt format and remaining runner
integration. New test fixtures are deliberately not paired benchmark artifacts.

## Tracking and publication

- [LatchMoE#94](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/94): parent stays
  open.
- [#96](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/96): bounded audit artifact
  delivery.
- [#97](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/97): offline gate candidate;
  live integration and review outstanding.
- [#98](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/98): model/TP2 resources and
  MOD qualification unresolved.
- [#99](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/99): actual five-dataset task
  quality unresolved.
- [#100](https://github.com/vLLM-HUST/vllm-ascend-hust-LatchMoE/issues/100): review/canonical merge
  and later website sync unresolved.
- [benchmark#254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254): canonical parent
  tracker.

This branch changes no canonical score/program executable state, global environment, website or old
evidence. PR submission/backlinks are authorized; automatic merges/deployment are not.
