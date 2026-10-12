# Qwen3.5-35B-A3B MMLU-Pro task accuracy (B0 development precheck)

- Model: Qwen3.5-35B-A3B, BF16, TP2 on 2x Ascend 910B2; full MMLU-Pro test set, 12,032 tasks.
- Accuracy: **9609 / 12032 = 79.86%**, with 56 invalid answers and 0 transport failures.
- Admission result: **development-phase precheck; not an admitted V5.5 B0 result; adds no Dataset
  Matrix cell; no B1 and no optimization claim.**

This report records one complete task-accuracy run and one throughput run on a development
container, with the exact commands, the frozen plan hashes, the startup receipt and per-task
outcomes. Positive, zero and negative observations are all kept. It does not change the frozen
Dataset Matrix contract or the program status of any dataset.

Tracking issue: #254 (MMLU-Pro: "run and publish task accuracy separately").

## Task accuracy

Strict extracted-answer accuracy; an answer that cannot be extracted, is outside the option range,
or belongs to a failed request counts as incorrect (the reference runner guesses randomly instead).

| Measurement                            |                    Value |
| -------------------------------------- | -----------------------: |
| Tasks                                  |                    12032 |
| Correct                                |                     9609 |
| Accuracy                               |                   79.86% |
| Invalid answers                        |                       56 |
| Transport failures                     |                        0 |
| Stopped by the 2048-token output limit |            1533 (12.74%) |
| Of those, still scored correct         |                      426 |
| Wall time of the run                   | 19800 s (concurrency 16) |

### Outputs that finished normally versus outputs cut off by the limit

| Group                | Tasks | Correct | Accuracy | Extraction stage 1 / 2 / 3 / none |
| -------------------- | ----: | ------: | -------: | --------------------------------- |
| Finished normally    | 10499 |    9183 |   87.47% | 10497 / 2 / 0 / 0                 |
| Hit the output limit |  1533 |     426 |   27.79% | 193 / 14 / 1287 / 39              |

Stage 3 takes the last standalone letter A-J anywhere in the text. For cut-off outputs it mostly
picks a letter that is ordinary prose (for example the article "A"), so those 'answers' are close to
noise. The overall accuracy includes them because the reference scoring does. The finished-normally
accuracy is shown as context only: easier questions finish sooner, so it is not an unbiased
estimate.

### By category

| Category         | Tasks | Correct | Accuracy |
| ---------------- | ----: | ------: | -------: |
| biology          |   717 |     663 |   92.47% |
| math             |  1351 |    1205 |   89.19% |
| economics        |   844 |     726 |   86.02% |
| physics          |  1299 |    1103 |   84.91% |
| business         |   789 |     669 |   84.79% |
| psychology       |   798 |     662 |   82.96% |
| chemistry        |  1132 |     939 |   82.95% |
| computer science |   410 |     339 |   82.68% |
| health           |   818 |     637 |   77.87% |
| other            |   924 |     717 |   77.60% |
| philosophy       |   499 |     379 |   75.95% |
| history          |   381 |     277 |   72.70% |
| engineering      |   969 |     624 |   64.40% |
| law              |  1101 |     669 |   60.76% |

### Tokens and client-side timing

- Prompt tokens: mean 1405, max 2910. Completion tokens: mean 736, median 494, p95 2048.
- Closed-loop client at concurrency 16: TTFT mean 0.59 s, p95 0.90 s; end-to-end mean 26.3 s, p95
  70.4 s, p99 82.1 s.
- These timings are not comparable with the throughput run below: its TTFT is dominated by queueing.

## Throughput run

200 tasks sampled at evenly spaced positions of the frozen plan (14 categories, 6-22 per category),
sent at once with `vllm bench serve` (`openai-chat`, `--request-rate inf`,
`--custom-output-len 256`, temperature 0, seed 0), the same flags as the other B0 workloads. The
prompts are the same text as the accuracy run; each was re-encoded and compared with the frozen
token IDs before use.

| Measurement              |                     Value |
| ------------------------ | ------------------------: |
| Completed / failed       |                   200 / 0 |
| Duration                 |                   115.6 s |
| Request throughput       |            1.73 request/s |
| Output token throughput  |             442.8 token/s |
| Total token throughput   |            2884.1 token/s |
| TTFT mean / median / p99 | 53476 / 55405 / 109507 ms |
| TPOT mean / median / p99 |  31.92 / 31.72 / 33.74 ms |
| Input / output tokens    |            282265 / 51200 |

Every request generated exactly 256 tokens (51200 = 200 x 256), so every request reached the output
limit. TTFT is large because 200 requests are submitted at once and the server runs 16 sequences at
a time. The throughput run goes through the chat endpoint with the server's default template
settings, while the accuracy run sends pre-rendered token IDs with thinking disabled, so output
lengths differ between the two. Single run; no repetition.

## What was run

| Item     | Value                                                                                                                                                                                                                                                                                                                                        |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model    | Qwen3.5-35B-A3B, local directory. Expected revision `59d61f3ce65a6d9863b86d2e96597125219dc754`; the four small files compared (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) match the SHA-256 values frozen in `reports/pyramidkv-five-dataset-adaptation-20261009`. Weight files were **not** hashed.   |
| Hardware | 2x Ascend 910B2 (logical devices 0,1)                                                                                                                                                                                                                                                                                                        |
| Engine   | vLLM `0.28.1.post1.dev143+gf18cf803c.empty`; vllm_ascend `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27.d20260909`; torch `2.13.0+cpu`; torch_npu `2.13.0rc1`; transformers `5.14.1`; Python 3.12.13; CANN 9.1.0                                                                                                                               |
| Server   | BF16, TP2 / PP1 / DP1, expert parallel on, effective DCP 1, `--disable-custom-all-reduce`, gpu_memory_utilization 0.85, max_model_len 32768, max_num_seqs 16, max_num_batched_tokens 8192, prefix caching off, chunked prefill on, no speculative decoding, not eager, `vllm_compile` mode 3, `FULL_DECODE_ONLY`, capture sizes 1 2 4 8 16   |
| Dataset  | TIGER-Lab/MMLU-Pro revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` (test and validation parquet SHA-256 checked by the adapter); reference runner revision `f418b116db00b065c2aea046518d8fcf74d39872`                                                                                                                                    |
| Plan     | frozen with `src/vllm_hust_benchmark/mmlu_pro_pyramidkv.py` of this repository (commit `9e7f83388d72a37767fd9f24636219093ecbe662`, SHA-256 `c2237c69f848e2358abe71c68f54143b5c090a81bd4f7915088b19b8ae157801`), `--per-category 5000` i.e. all 12,032 tasks; `cases.json` `f77a7e5e6fd96fbbf46be7ab11f360309554105e97921805f02d1c93fc05fcf3` |
| Requests | `/v1/completions` with the frozen token IDs, streamed; temperature 0, seed 17, thinking disabled (in the rendered prompt), max_tokens 2048, no stop string, no retries, concurrency 16                                                                                                                                                       |
| Scoring  | the adapter's `score_results`, unchanged                                                                                                                                                                                                                                                                                                     |

Startup receipt: 18 of 18 expected configuration fields are confirmed by both the command line and
the server startup log (`receipt/effective.reparse.json`). The first comparison made at launch
(`receipt/effective.json`, older log parser) confirmed 13 fields and could only confirm 5 from the
command line; the later re-parse of the same frozen startup log is kept next to it and the original
is not rewritten. The actual command line captured from `/proc` equals the planned one except for
the interpreter wrapper in front.

The accuracy run used the adapter for planning and scoring but its own client for transport
(`reproduction/mmlu_pro_run.py.txt`, SHA-256
`5d647fee41d599830f6bab705e4c36cff712b72d89f29e6d898c6d3122448c6f`, equal to `runner_sha256` in
`results/accuracy-run-meta.json`), because the adapter's `run` command validates KV-compression
activation of the server, which a plain B0 server does not have. The compression plugin was not
loaded (no `kvcompress` plugin registered, no `VLLM_ASCEND_KVCOMPRESS_*` variable, no mention in the
server log), and all prompts are 898-2,910 tokens, below the 4,096-token compression threshold.

## Contract boundary

This report does **not** claim:

- A formal V5.5 B0 result. The engine is not the pinned baseline (V5.5 names upstream vLLM 0.18,
  core `bcf2be96...`, plugin `e18643f8...`); there is one lifecycle instead of three; the client
  contract differs (V5.5's reasoning cell uses chat completions, a 512-token output cap and Poisson
  arrivals); the evidence is not signed; the container image digest was not available
  (`NOT_PROVIDED`).
- A Dataset Matrix cell, a change of any program status, or a leaderboard entry.
- Any B1 comparison, speedup or optimization benefit. There is no B1 run.
- The upstream leaderboard configuration of MMLU-Pro: prompts are wrapped in the chat template with
  thinking disabled, all five shots are kept, there is no `Question:` stop string, the context is
  32K, and a task whose answer cannot be extracted is incorrect instead of randomly guessed.
- Run-to-run stability. See the next section.

## Observations

1. **Outputs were not reproducible run to run.** For the same 5 questions, two further runs of the
   same client and one run of a second client gave pairwise different texts (0 of 5 identical,
   completion lengths such as 351, 340 and 652 tokens) at temperature 0 with a fixed seed, and also
   differed from the full run. This was seen once, on 5 questions, under concurrent load; it is not
   a measurement of the size of the effect. A single run's accuracy therefore carries unestimated
   noise.
1. 12.74% of the tasks stop at the 2048-token limit and score 27.79% (see above); a lower output cap
   would push more tasks into that group.
1. Law (669/1101) and engineering are the weakest categories; biology and math the strongest.

## Evidence grading

| Grade          | Content                                                                                                                                                                                                                                                                                                                                                                                                   |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R (observed)   | Every raw request/response record (12,032) is hashed in `index/raw-request-files.sha256.gz`; the per-task table recomputes the totals; the throughput result JSON is hashed next to the command that produced it; the startup log snapshot and `/v1/models` response are in `receipt/` and the raw bundle                                                                                                 |
| D (documented) | Intended configuration: `reproduction/b0_35b.profile.json`                                                                                                                                                                                                                                                                                                                                                |
| S (snapshot)   | Package versions, environment variables, `npu-smi` taken at launch (`raw/run-evidence.tar.gz`)                                                                                                                                                                                                                                                                                                            |
| R (sealed)     | After the server was stopped the run directory was sealed as a plain SHA-256 listing: 12,091 files, `evidence_bundle_sha256` `ee2005782b49a4b228cb64386c3fb94892f7d9aede2da0745e349e8680d78caf`. `index/evidence-bundle.json.gz` lists every file's SHA-256 and size; its entries for `receipt/host.txt` and `receipt/versions.txt` are the hashes of the unredacted originals. Not RFC 8785, not signed. |
| X (missing)    | Container image digest; full weight hashes; a signed manifest, or a `MANIFEST.json` binding a results table (no results table exists, so only the evidence listing was sealed); B1                                                                                                                                                                                                                        |

## Open questions for maintainers

1. If this should become a Dataset Matrix cell, it needs a task-accuracy metric (id, unit,
   direction) and an evidence grade for the run binding; none is defined for this artifact yet.
1. The existing Qwen3.5 B0 scenario in `leaderboard-data/dataset-validation` records model revision
   `712cf74392b05026a6db2bf213d343747d1f6d45`, while V5.5 table 附-8 and this report use
   `59d61f3c...`.
1. Whether an unsigned development precheck may be shown in the matrix at all, given the V5.5 rule
   that only signature-verified bundles enter the result snapshot and the website.

## Files

| Path                                                                  | Content                                                                                                                                                                     |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results/accuracy-summary.json`                                       | Full accuracy report (totals, termination split, categories, tokens, timing, run metadata)                                                                                  |
| `results/accuracy-run-summary.json`, `results/accuracy-run-meta.json` | The accuracy runner's own summary and run record                                                                                                                            |
| `results/per-case-outcomes.csv.gz`                                    | One row per task: id, category, gold letter, prediction, valid, correct, finish reason, token counts. No question or model text                                             |
| `bench/`                                                              | Throughput result JSON, summary, the 200 sampled case ids                                                                                                                   |
| `plan/`                                                               | Frozen plan contract, runtime contract, prepare log                                                                                                                         |
| `model/qwen35-small-files.json`                                       | SHA-256 of the model's small files                                                                                                                                          |
| `receipt/`                                                            | Startup receipt comparisons and `/v1/models` (readable copies; a final newline was added)                                                                                   |
| `raw/run-evidence.tar.gz`                                             | Byte-exact originals of the receipt, run records, logs and bench files (see Redactions)                                                                                     |
| `index/raw-request-files.sha256.gz`                                   | SHA-256 of every raw request/response file (kept on the development container, not published: model text restates source questions)                                         |
| `index/excluded-large-files.txt`                                      | SHA-256 and size of files too large for this repository                                                                                                                     |
| `index/evidence-bundle.json.gz`                                       | The sealed listing of the whole run directory (12,091 files: path, SHA-256, size)                                                                                           |
| `reproduction/`                                                       | Launcher, parser, accuracy client and summary script exactly as run (`.txt`), the intended-configuration profile, and the hashes of the modules used for the throughput run |

## Redactions

The container hostname (it contains a user name) was replaced by `<pod-hostname-redacted>` in
`receipt/host.txt` and `receipt/versions.txt` inside `raw/run-evidence.tar.gz`. The bundle was
rebuilt deterministically; its SHA-256 before the redaction was
`a52c3427f824277d620fa84db74193d214a0ed4ed14e00aa8f2e6895537edae3` and the published one is listed
in `SHA256SUMS`.

## Verify

```bash
cd reports/mmlu-pro-qwen35-b0-precheck-20261012
sha256sum -c SHA256SUMS
gunzip -c results/per-case-outcomes.csv.gz | awk -F, 'NR>1{n++; c+=$6; i+=(1-$5); l+=($8=="length")} END{print n, c, i, l}'
# expected: 12032 9609 56 1533
gunzip -c index/evidence-bundle.json.gz | python3 -c "import json,sys,hashlib; b=json.load(sys.stdin); h=hashlib.sha256(); [h.update(('%s  %s\\n'%(b['files'][r]['sha256'],r)).encode()) for r in sorted(b['files'])]; print(len(b['files']), h.hexdigest())"
# expected: 12091 ee2005782b49a4b228cb64386c3fb94892f7d9aede2da0745e349e8680d78caf
```
