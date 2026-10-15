# HLE-Verified: Qwen3.5-35B-A3B, Gold + Revision

Development evaluation, **not official leaderboard replication or V5.5 acceptance**. This report
does not update official baselines, leaderboard snapshots, or paired B1 comparisons.

| Subset   | Correct / selected | Accuracy | Unresolved judge | Solver length | Judge length |
| -------- | -----------------: | -------: | ---------------: | ------------: | -----------: |
| Gold     |          156 / 668 |   23.35% |               32 |           383 |           32 |
| Revision |         313 / 1143 |   27.38% |               40 |           688 |           40 |

The author-published HLE-Verified dataset is pinned in `report.json`; Uncertain 689 is excluded.
Original HLE system/judge prompts and JSON schema are used with the same Qwen model as solver and
substitute judge, one trial. Strict JSON grading requires a normally completed response. All 72
length-ended judges remain **unresolved**, in the full denominator; they are not confirmed
incorrect. No partial JSON rescue or retries. Solver truncations (1071/1811) are sent to judge, not
automatically marked wrong. This large truncation count materially limits interpretation.

Solver/judge concurrency: 16/16; max_tokens: 16384/8192; temperature0, seed17, thinking=false.
Source versions, dataset revision, full scoring contract and per-case outcome hashes are included.
The HLE client remains a local project; this publication does not create a client GitHub repository.

## Separate vLLM bench serve runs

| Group          | Completed | Failed | Output token/s | Mean TTFT s | Mean TPOT ms |
| -------------- | --------: | -----: | -------------: | ----------: | -----------: |
| Gold text      |       200 |      0 |        381.323 |      57.849 |       38.784 |
| Revision text  |       200 |      0 |        422.899 |      52.609 |       36.822 |
| Gold image     |        93 |      0 |        314.484 |      28.901 |       52.213 |
| Revision image |       118 |      0 |        330.824 |      34.362 |       42.605 |

Text selection: `floor(i*N/200)` after modality filtering in frozen task order. Image subsets are
complete and retain original question text and image bytes; no repetitions or correctness filtering.
Inputs, row IDs, text hashes and decoded image byte hashes are bound to the accuracy contracts.

All four groups use one user message, no accuracy system prompt, server template once,
thinking=false, temperature0, request/dataset seed0, max_completion_tokens256 and ignore_eos=false.
EOS may end early. request-rate=inf and no client max-concurrency; server max_num_seqs16 is
different from fixed client concurrency16. TTFT includes burst queueing. Each vLLM run sends one
initial test request outside the measured N; additional warmups=0. Service is reused after accuracy,
image smoke and tokenization preflight, not cold-cache. No reruns or speedup claim.

All 611 per-request input lengths matched same-server `/tokenize`, including images. Installed
openai-chat consumes streaming usage; no separate visual-token count is claimed. Full raw results
remain in the original sealed bench directory; public summaries are linked by original file hashes.

## Runtime and provenance

The archived actual argv, startup log and effective receipt verify 20 fields: BF16; TP2/PP1/DP1, EP
enabled/DCP1; disable_custom_all_reduce=true; gpu_memory_utilization0.85; max_model_len65536;
max_num_seqs16; max_num_batched_tokens8192; prefix=false, chunked=true, speculative=None;
eager=false, compilation mode3, FULL_DECODE_ONLY, capture sizes1/2/4/8/16; xgrammar with
disable_any_whitespace=true. ASCEND_RT_VISIBLE_DEVICES=0,1. Hardware/version details come from
archived npu-smi and package receipts, not the official v0.18 baseline.

Model small files were hashed and compared; full weight shards were only inventoried. A verified
local model revision and container image digest are unavailable. These limitations are not repaired
by assuming the profile's expected revision is the actual artifact.

There are **three independent original seals**, for Gold, Revision and bench. `report.json` and
`publication.json` preserve all table/bundle/manifest hashes. Each `*-evidence.tar.gz` includes the
original full evidence index and selected execution evidence. Original manifests and result tables
remain byte-identical; operator identity is redacted only where recorded in the publication mapping.
Raw questions, images, solver/judge answers and environment dumps are excluded. Thus public checks
validate reported outcomes and evidence binding but cannot independently regrade answers.

```bash
python reports/hle-verified-qwen35-20261015/verify_public.py
```

To regenerate from private originals, use
`export_public.py ACCURACY_CAMPAIGN BENCH_RUN OUTPUT --client-root LOCAL_HLE_CLIENT`, then run the
public verifier. Keep outputs outside sealed runs. `SHA256SUMS` covers all published files other
than itself. SHA-256 is integrity checking, not a signature. No merge or automatic deployment is
authorized by this report.
