# BetterScale + FA Microflow: warm C32 MMLU throughput

Measured 2026-10-20 on one node, 2 × Ascend 910B2 (physical 2/4,
64 GiB each). Four cards reserved to isolate the host; only two execute.
All cards idle on release. This is a throughput experiment, not an accuracy
submission or an official fixed-target leaderboard entry. No same-spec hash
is borrowed from another deployment.

| Burst | Role | Output token/s | Completed / failed | Compiler-cache changes |
|---|---|---:|---|---|
| 0 | cold warmup | 464.996951 | 200 / 0 | yes |
| 1 | warmup | 827.699461 | 200 / 0 | one additional `correct` specialization |
| 2 | formal | 882.093459 | 200 / 0 | none |
| 3 | formal | 888.082845 | 200 / 0 | none |
| 4 | formal | 873.941535 | 200 / 0 | none |

**Formal median 882.093459; mean 881.372613 output token/s.** Each burst:
282,265 input and 51,200 output tokens. All five runs retained; no outlier removal.
One server start, two full-workload warmups, three timed repeats. Cache audit
compares file size/mtime before/after each burst, not every in-memory compiler event.
Prefix-hit logs during the observed bursts remain 0.0%; no cache-salt plugin.

## Deployment and workload

- Qwen/Qwen3.5-35B-A3B, real BF16 weights; TP2, MTP2, FULL graphs, APC,
  async scheduling, AIV collective. Execution C32 / resident R36; max context
  65,536; batch budget 4,096; State20 GiB and Host8 GiB per rank.
- BetterScale `2c07ddd5eb04f2afeb77123ce066a3a70662a94c`;
  sealed school FA Microflow `f7ab240381a0e10a4a13f4ee04598cb822db6fdd`.
  This is the school FA-prefix adapter, **not the newer solver graph pool**.
- vLLM `0.28.1.post1.dev143+gf18cf803c.empty`, Ascend
  `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`, Torch2.13 CPU,
  torch-npu2.13rc1, Transformers5.14.1, CANN9.1.
- Original school-qualified GDN host library, not tutor's rebuilt artifact;
  [explicit host pin delta](host-artifact-pin-delta.json). Other checked
  adapter host-source pins retained. No installed runtime changes.
- Same original 200 MMLU prompts/client as the midterm report's FA on-r9 arm.
  Materialized JSONL SHA256:
  `fb79b6fa98f232282a948e4a17f404f80c03dc2bae05af57b95b9cc0651b3506`.
  No truncation or alternate prompt format introduced.
- Client: `vllm bench serve --backend openai-chat --endpoint /v1/chat/completions
  --dataset-name custom --disable-shuffle --custom-output-len 256 --num-prompts 200
  --request-rate inf --temperature 0 --seed 0 --save-result --save-detailed
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99`.
  Model alias `qwen3.5-35b`; tokenizer is the same model. Server default
  thinking=true. C32 is server execution-seat capacity, not client max-concurrency.
  [Effective server argv](server-argv.json).
- Both ranks: exact partition equivalence at 1K/2K/4K; active FA adapter;
  360 capturing prefix events and 6,994 graph replay events each. GDN and draft
  unchanged by FA adapter. No benchmark accuracy score inferred.

## Interpretation

The warm median is 6.934% above the prior C24 FA median824.894609 and 84.311%
above historical C32 BetterScale478.590799. These are **cross-run references**,
not paired causal gains: C24 used three independent starts; this experiment
uses one, and the native host binary difference is explicit. The old C32 did
not have this FA configuration either. This run directly demonstrates cold/warm
sensitivity; it does not prove the precise cause of the historical C32 result.
Do not replace historical measurements or report the cold465 as warm C32 capacity.

## Evidence and reproduction

Payload-free client JSON files retain all numeric metrics and per-request timing
arrays. Only generated texts/errors are omitted; all error counts are zero.
[Archive/raw bindings](evidence-binding.json) bind the private full-response
archive. The archive is retained by the experiment owner, not uploaded here.
[All measurements and cache changes](measurements.json).

Harness and detailed deployment notes:
[BetterScale experiment branch](https://github.com/vLLM-HUST/BetterScale/tree/ffc09817/prototypes/microflow-c32-check).
