# School-runtime acceptance — v1

On 2026-10-16, BetterScale completed the BetterScale column's evaluation scopes from
[dataset matrix v1](https://vllm-hust.sage.org.ai/dataset-matrix-v1.html) using the
school-maintained runtime. The school environment is an additional qualified donor, not a separate
inference implementation. Minimal interface bridges preserve MTP, FULL graphs, State/cache
management and numerical kernels.

## Environment and compatibility

- Qwen3.5-35B-A3B BF16, TP2/MTP2, FULL graphs, maximum context 65,536, query budget 4,096, State 20
  GiB/rank and host cache 8 GiB/rank.
- Four Ascend 910B2 cards; two concurrent TP2 lanes. Text used E16/R36; multimodal used E16/R20 with
  up to four images per request.
- Python 3.12.13; torch 2.13.0+cpu; torch-npu 2.13.0rc1; transformers 5.14.1; CANN 9.1.0.
- vLLM `0.28.1.post1.dev143+gf18cf803c.empty` and vLLM-Ascend
  `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909`. Exact source admission is recorded in
  `qwen35_school_pins.json` and `models/qwen35/runtime-school.json`, not inferred from version
  strings alone.

The bridges cover draft sampling metadata/return shape, Mamba postprocessing, strict configuration
extension keys, already-published KV view registration, scheduler source seams and the school
prefix-query tuple. Existing donor routes remain intact. The unchanged GDN host adapter was rebuilt
against the school headers; GDN/FIA device kernels were not replaced. The installed school sources
were not modified: execution used an isolated donor runtime.

## Accuracy

| Scope                    | Correct / total | Accuracy | Previous matrix correct / total |
| ------------------------ | --------------: | -------: | ------------------------------: |
| MMLU-Pro                 |  9,589 / 12,032 |   79.70% |                  9,611 / 12,032 |
| FrontierScience Olympiad |        60 / 100 |   60.00% |                        63 / 100 |
| FrontierScience Research |          3 / 60 |    5.00% |                          2 / 60 |
| HLE-Verified Gold        |       173 / 668 |   25.90% |                       168 / 668 |
| HLE-Verified Revision    |     316 / 1,143 |   27.65% |                     320 / 1,143 |

Gold text/image counts were 150/575 and 23/93; Revision text/image counts were 299/1,025 and 17/118.
Solver and judge API failures were zero. HLE unresolved judgments (18 Gold, 28 Revision) and
Frontier unresolved judgments (3 per subset) remain in the full denominators; no retries or
selective exclusions. MMLU had 57 invalid extracted answers. Output caps were retained, not
repaired: 1,523 MMLU, 380 Gold and 674 Revision solver responses reached the length limit.

The frozen clients, prompts, data and scoring conventions match the previous matrix: MMLU 5-shot,
concurrency 16 and 2,048 output tokens; Frontier solver/ judge concurrency 1 and 16,384/8,192
tokens; HLE concurrency 16 and 16,384/8,192 tokens with strict JSON judging. Temperature 0, seed 17
and thinking disabled were retained. Frontier uses the original `mmlu_extract_first` judge handling;
HLE uses the recovered client used by the earlier matrix. These are same-model-judge matrix reruns,
not an official leaderboard replication.

## Performance

Separate burst tests use the original 256-output-token cap and unlimited arrival rate. These rates
are not accuracy-run throughput or SLO goodput.

| Scope          | Requests | Output token/s | Previous matrix output token/s |
| -------------- | -------: | -------------: | -----------------------------: |
| MMLU           |      200 |         631.59 |                         624.82 |
| Frontier       |      160 |         906.25 |                         922.46 |
| Gold text      |      200 |         875.67 |                         916.40 |
| Revision text  |      200 |         871.63 |                         918.63 |
| Gold image     |       93 |         489.94 |                         504.10 |
| Revision image |      118 |         466.19 |                         460.18 |

All requested counts completed with zero failures. All four HLE benchmark input token counts matched
the server's `/tokenize` accounting. These are single runs with overlapping lanes; small differences
are not attributed to a particular runtime change. No further performance sweep or causal study is
part of this acceptance.

## Validation and evidence

The school Python passed 18 CPU interface/fairness tests and the loaded-platform scheduler AST
check. A CPU fixture exercised the actual school scheduler's cold/warm admission and decode/prefill
contention. Device smoke captured all 32 FULL graphs and passed 23/23 text/image checks. Mixed C16
efficiency smoke completed 16/16 requests at 325.31 output token/s; text smoke completed 16/16 at
418.43. Four HLE protocol smoke cases passed HTTP/schema checks, but scored 0/4: protocol success
was not treated as evidence of answer correctness.

Full counts, runtime/source identities, per-request outputs, failed startup receipts, launchers and
service release evidence are retained in the private
`lab-dataset-acceptance-20261016/acceptance-evidence.tgz` capsule under the BetterScale external
evidence archive. `report.json` contains the validated aggregate, and `acceptance-runtime.json`
freezes the exact tested code/config. Raw datasets, images and responses are not committed to this
repository.

The release tag is `v1`, with package metadata version `1.0.0`; this is a Git source release, not a
PyPI upload. Relative to the frozen tested overlay on `7740df9`, final changes are documentation,
package version and qualification metadata only. Both service lanes completed normally and released
all four cards. Unrelated in-progress cache/profiling changes are excluded from v1.

## Public projection

`report.json` publishes aggregate quality, six independent load tests, exact runtime versions and
integrity bindings to the private evidence capsule. It contains no questions, images, raw responses
or connection information. Historical reports are unchanged. This compatibility acceptance does not
grant formal fixed-target leaderboard admission.

Regenerate from the retained private capsule with:

```bash
python reports/betterscale-school-runtime-20261016/export_public.py \
  PRIVATE_CAPSULE.tgz PRIVATE_RELEASE.json \
  reports/betterscale-school-runtime-20261016/report.json
python reports/betterscale-school-runtime-20261016/verify_public.py
```

The verifier checks aggregate consistency and public-file integrity, not independent regrading. The
immutable implementation and detailed compatibility report are in
[BetterScale v1](https://github.com/vLLM-HUST/BetterScale/tree/v1).
