# Runtime MOD review and bounded reproduction — 2026-10-27

Reviewer: Fletcher. This collection contains the Chinese review report, independent 35B vSpec / 27B
DiffSpec screens, OPset operator probes, frozen evidence and an offline pre-publication audit. It is
**not an official leaderboard submission, full MMLU-Pro evaluation, V5.5 acceptance, or proof that
all reviewed MODs are production-qualified**. No leaderboard or website metrics are changed.

## Read first

- [Complete Chinese report](research/report.md) · [PDF](output/pdf/runtime-mods-review.pdf)
- [Formal review table](research/comments.md): adoption judgments are distinguished from
  measurements.
- [35B vSpec results and system-difference table](research/speculative-mmlu-pro/results-qwen35.md)
- [Pre-publication audit](research/speculative-mmlu-pro/publication-audit.md)
- [DiffSpec valid short comparison and incomplete scope](research/speculative-mmlu-pro/results-diffspec.md)
- [OPset two-card operator diagnostic](research/opset/reproduction/README.md)
- [Recomputed numeric results](recomputed-results.json)

## What the evidence supports

The latest matched-depth 35B comparison uses native MTP gamma4 versus the original vSpec
configuration, which actually stayed at gamma4 with zero switches in all three runs. The same 56
questions were repeated in fresh engines three times per arm. Effective gamma4 target FULL graph
keys matched on both ranks before and after all six runs. Median output throughput was **811.05
tokens/s native versus 795.60 tokens/s vSpec** (candidate/native 0.98095). No incremental throughput
benefit was observed for this 35B/B16 workload. Other runtime paths and shared-host interference
remain; this is not a claim of statistically significant slowdown or a rejection of adaptive
speculation in general.

[Matched-gamma4 protocol, results and verifier](research/speculative-mmlu-pro/gamma4/RESULTS.md) are
the current performance evidence. The concise report contains this experiment only. Historical
280-question scores (203/202/193) and the old 1.17243 wall ratio against native MTP2 remain in the
research notes, not as a matched-baseline advantage or causal quality-loss proof. All four original
35B candidate runs also stayed at gamma4: configured candidates 2/4/6 were not observed dynamic
switching. Both MTP arms used FULL_AND_PIECEWISE; no extra graph optimization has been isolated as a
cause of the old gain. Acceptance relaxation was disabled.

[Follow-up diagnostics](research/speculative-mmlu-pro/divergence/RESULTS.md) retain
first-divergence, norm and bucket observations. Synchronous diagnostic timings are not performance
evidence; the corrected-norm candidate was stopped, not a completed three-arm quality experiment.

DiffSpec's valid C04/B16 three-arm 56-question first pass had no preemptions and no speed gain:
232.12 / 95.58 / 91.34 output tokens/s for target-only / Eagle / DiffSpec. Default linear retrieval
did not execute. Full three-arm 280 questions and performance repeats 2/3 were not completed;
additional experiments were explicitly stopped. Degenerate and preempted earlier runs are retained
but excluded from algorithm inference.

OPset's fused operator took 48.37%–82.31% more wall time than ordinary ACLNN separation across two
cards and three shapes. Percentages are recomputed from unrounded samples, superseding 48.36%–82.30%
calculated from rounded display values. This is an eager synchronized **operator diagnostic**, never
a graph-serving leaderboard result. Unnecessary zero-initialization removal is a separate
small-patch recommendation.

## Provenance and limits

35B / DiffSpec host: Ascend 910B2; vLLM Core `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`, Ascend
`5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`; BF16, TP2 for 35B and TP4 for DiffSpec. vSpec source
`4e17b6bb6f9957dd4427f112da0bfc3d08bf24ae`; DiffSpec source
`1294c95b6d9b2d9cbed0d0cffe31202fe5a6a254`. OPset used a different recorded Ascend revision; see its
own protocol. No cross-model accuracy comparison is claimed.

Dataset: TIGER-AI-Lab/MMLU-Pro revision `b189ec765aa7ed75c8acfea42df31fdae71f97be`; subject-balanced
20×14 screen and a frozen 4×14 performance subset, 5-shot CoT, temperature 0, max output 4000. The
input files, parser and per-request generated text/token IDs are archived. Source-level input
preparation and matching IDs/input lengths are checked; no per-run historical prompt-byte hashes
were recorded. Target weight identity is local config/index/size/mtime, not a verified immutable
upstream weight revision. No new NPU measurements were made during the publication audit.

The archive includes failures, diagnostics and superseded 14B runs for custody, **not as accepted
results**. The allowlist in `verify_results.py` selects only 12 formal 35B runs plus 3 valid
DiffSpec runs, and separately aggregates the OPset operator evidence. Some original run identifiers
contain the assistant's historical workspace prefix; these immutable provenance identifiers are not
report authorship.

## Verify and reproduce the analysis, without hardware

```sh
cd reports/runtime-mod-review-20261027
python verify_archive.py
python verify_archive.py --restore .materialized
python verify_results.py .materialized > .materialized/recomputed-results.json
python .materialized/research/speculative-mmlu-pro/gamma4/analyze.py > .materialized/gamma4-recomputed.json
```

The archive is split into ordered parts below 1 MB. `raw/manifest.json` binds each part, complete
gzip archive and every original member. The restorer verifies digests and regular-file relative
paths, refusing conflicting overwrites. It does not execute archived code. `verify_results.py` then
runs only the frozen answer parser and analysis functions: it does not load a model or invoke
accelerator libraries. Compare its JSON structurally with `recomputed-results.json`; JSON key
ordering/formatting is not measurement identity. Compare the gamma4 output structurally with
`research/speculative-mmlu-pro/gamma4/results.json`; that independent verifier validates the six new
runs, answer parsing, actual gamma decisions and per-rank graph keys.

Public JSONs outside the raw capsule are readable copies/derived summaries. The capsule, not
formatting of those copies, owns original byte identity. Readable operator producer copies use
`.py.source`; exact runnable producers, inputs, configurations, logs and source excerpts are
recovered under `.materialized/`. File links into archived raw evidence may require restoration;
they are not missing experiments.

The vSpec executed source recovered during audit differs from pinned upstream only by the saved
two-hunk statistics/empty-step patch. Selected host source is also archived, with its license. The
private bootstrap adapter is disclosed separately. This retrospective comparison does not invent a
source snapshot for every historical run.

## Rebuild the report

The PDF generator needs ReportLab and the macOS STHeiti fonts. The delivered PDF embeds its CJK
fonts. Repository-facing links use the published review-branch report location, not local `file:///`
paths:

```sh
REPORT_LINK_BASE=https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/review/runtime-mods-evidence-20261027/reports/runtime-mod-review-20261027/research/ python research/build_report.py
```

Research notes retain narrower evidence boundaries and original source citations; they are not all
independent replications. Later review changes should be versioned without overwriting this
capsule's measurements.
