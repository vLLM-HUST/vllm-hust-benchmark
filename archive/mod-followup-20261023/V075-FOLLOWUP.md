# V0.75 custody follow-up — 2026-10-23

This is a new dated observation, not a replacement for the original [batch receipt](README.md). The
[navigation audit](../../docs/data-entrypoints-v075-audit.json) reconciles the deployed website and
Release entries. No raw experiment, source duplicate or installation asset was changed here.

## Migration inventory

The fresh [public inventory](v075-public-inventory.json) covers 66 public non-archived repositories,
4,156 candidate Git paths and 27 Release asset records. Compared with `public-inventory.json`, the
only difference is the 13 BiDKV copies already removed by BiDKV #24. All other candidate blob IDs
and asset metadata are unchanged. These are heuristic candidates, not measurement counts or deletion
permission. Private repositories are excluded. The policy script at
`vLLM-HUST/.github@cb8273cb6e13345f8f8635ffb7b9647c4f4035fa` is byte-identical to the current merged
script at `6ff1a382ea39a86ad4fe8b9c9928ab5496603631`.

| Remaining queue                    | Candidate paths | Required next step / owner                                                            |
| ---------------------------------- | --------------: | ------------------------------------------------------------------------------------- |
| KV materialization arrival control |            1399 | MOD maintainers: trace WaveMat/raw-result consumers before moving originals           |
| PegaFlow                           |             777 | MOD maintainers: classify issue23 handoff and replay consumers                        |
| MuxPD                              |             562 | MOD maintainers: distinguish TP4 measurements from implementation/graph documentation |
| Request lifecycle profiler         |             441 | MOD maintainers: separate measured captures and synthetic causality fixtures          |
| SimLLM                             |             145 | MOD maintainers: match archived website originals and migrate remaining consumers     |
| PyramidKV                          |             123 | MOD maintainers: preserve run/model/dataset identity and failures                     |
| KVCompress                         |              73 | MOD maintainers: trace raw-result consumers                                           |
| vSpec                              |              57 | MOD maintainers: inspect remaining paper/result dependencies                          |
| Extension Manager                  |              42 | MOD maintainers: separate host contracts from measured qualification evidence         |

The 64 Split-Batch candidates also include implementation/reference paths; they do not mean 64
unmigrated measurements. Its 12 Release asset records are unchanged. The previously verified
PyPI/installation consumers still prevent a zero-consumer deletion conclusion. No installation
distribution or checksum was deleted. The prior 33 missing probes remain documented as lost; search
and the cancelled GDN retest were not resumed.

## External gates re-read, not bypassed

| PR                                                                      | Re-observed state                                                                                                                                                                                | Required action / owner                                                                            |
| ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| [Triton #10](https://github.com/vLLM-HUST/triton-ascend-hust/pull/10)   | OPEN, head `42b416f9a2ca74779ef401e078bc0f9d8af9b02f`; review job 119207435721 cancelled, requested `linux-amd64-cpu-8-hk`, runner name empty; latest title check passed, older failure retained | Repository runner owner restores/authorizes review capacity, then reruns the actual review         |
| [LMCache-Ascend #1](https://github.com/vLLM-HUST/LMCache-Ascend/pull/1) | OPEN, head `c8d83aa2eb94e260ec0c73f7b64b22ed4873e72f`; 910B Unit, 910C Unit, 910C Perf & Accuracy remain PENDING                                                                                 | Repository/device owners supply actual required hardware checks                                    |
| [SGLang #2](https://github.com/vLLM-HUST/sglang-hust/pull/2)            | OPEN, head `e842fbef8e4a9e666023a1549412a8da4caf1a26`; labels empty, extra gates/finalizers still FAILURE                                                                                        | Current maintainers approve the appropriate CI before required device checks; no labels added here |

CPU source/policy passes do not waive these gates. No forced merge, rerun, approval label or device
job was requested. #326 stays open for these queues; this audit does not claim new MOD migrations.

Skill review: current request is index/Release and migration/gate reconciliation; live website
manifest, paginated Release API, public Git fingerprints and PR checks are the evidence.
E01/E04/E31/E32/E37/E39/E42/E45: repair stale metadata, retain original failures and fixed B0/B1, do
not turn an inventory match into measurement acceptance or deletion authority.
