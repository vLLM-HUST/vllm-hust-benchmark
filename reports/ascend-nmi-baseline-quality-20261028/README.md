# Natural-generation and independent native A/A controls

Research evidence from executed Ascend 910B2 experiments on the user-owned evaluation container. These are BF16 research controls, not official leaderboard submissions. Original finite regressions, warmups, source identities and process logs are retained.

## Design and estimands

Twelve independent processes: three blocks, each with native setting0 twice (native0a/native0b), native setting1, and conditional setting0. Six reused cohorts contain 28 news and 24 report contexts, with 32 requests each. One warmup cohort per process is excluded; 2,304 formal answers use greedy natural termination and a 2,048-token cap. No forced-output logits processor is installed. Common synchronous instrumentation is retained; this experiment does not estimate HTTP latency or fixed-work speedup.

All full token sequences and finish reasons are retained. Complete decoded responses are scored with the unchanged ROUGE-L F1 implementation. One question corresponds to each exact context. Repeated appearances are averaged within context/block, then equally across three blocks and unique contexts. Percentile intervals use 5,000 fixed-seed context resamples, conditional on the observed processes. They are exploratory intervals, not noninferiority tests, factuality judgments, or independent-document replication. Native A/A variation covers this synchronous implementation, not all arrival-induced service variation.

## Comparisons

Positive score differences favour the candidate; caps are baseline/candidate. Sequence changes count paired full-answer differences.

| Baseline | Candidate | Dataset | ROUGE-L delta (pp) | Context bootstrap 95% (pp) | Sequence changes | Caps |
|---|---|---|---:|---:|---:|---:|
| native0a | native0b | gov_report | +0.0000 | +0.0000 to +0.0000 | 0/72 | 0/0 |
| native0a | native0b | multi_news | +0.0000 | +0.0000 to +0.0000 | 0/504 | 0/0 |
| native0a | native1 | gov_report | -0.0355 | -1.0431 to +1.0866 | 72/72 | 0/0 |
| native0a | native1 | multi_news | -0.3936 | -1.0404 to +0.2297 | 405/504 | 0/0 |
| native0a | conditional0 | gov_report | -0.3828 | -1.1034 to +0.3397 | 72/72 | 0/0 |
| native0a | conditional0 | multi_news | -0.2991 | -0.9656 to +0.3324 | 405/504 | 0/0 |
| native1 | conditional0 | gov_report | -0.3473 | -1.2616 to +0.5196 | 72/72 | 0/0 |
| native1 | conditional0 | multi_news | +0.0945 | -0.4150 to +0.5973 | 405/504 | 0/0 |

CPU rescoring dependencies: `rouge==1.0.1`, `transformers==5.5.4`. The exact tokenizer files are included; model weight files are not. `scored-answers.json` retains every decoded response and score.

## Verify and reproduce

`manifest.json` binds every byte of the ordered base64 archive parts, all archive members and derived public files. The capsule contains frozen plans, source snapshots, raw result JSON, logs, warmups and executable analysis. Verification does not execute archive content.

```sh
python verify_archive.py --extract /tmp/ascend-nmi-baseline-quality-20261028
python /tmp/ascend-nmi-baseline-quality-20261028/campaign/analyze.py
```

The archived scripts retain original absolute runtime paths as provenance. New device execution requires the recorded model/runtime, a free device lease and explicitly rebased paths/hashes; it must write a fresh directory and never overwrite this evidence. No model weights are uploaded. This collection does not complete the public release of all original study data.

For CPU quality rescoring, use the dependency versions listed above in an isolated environment.
