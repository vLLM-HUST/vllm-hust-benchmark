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

The 35B arms scored 203/280, 202/280 and 193/280. The adaptive whole-configuration wall-time ratios
were 1.53167 versus target-only and 1.17243 versus native MTP2. Both MTP arms used
FULL_AND_PIECEWISE graphs. Gamma, refill, capture buckets, dynamic-width runtime support and private
compatibility code differ: this is a whole-configuration comparison, **not a gamma-only ablation or
proof that additional graph optimization caused the gain**. Confidence-margin acceptance was
disabled; the unrelated historical 14B 1.518x result is not this measurement. Quality causality is
unresolved under truncation and native run variability; current adoption is rejected, not the
measured whole-stack speed difference.

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
```

The archive is split into ordered parts below 1 MB. `raw/manifest.json` binds each part, complete
gzip archive and every original member. The restorer verifies digests and regular-file relative
paths, refusing conflicting overwrites. It does not execute archived code. `verify_results.py` then
runs only the frozen answer parser and analysis functions: it does not load a model or invoke
accelerator libraries. Compare its JSON structurally with `recomputed-results.json`; JSON key
ordering/formatting is not measurement identity.

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
fonts. Repository-facing links use the public main-branch report location (available after merge),
not local `file:///` paths:

```sh
REPORT_LINK_BASE=https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/main/reports/runtime-mod-review-20261027/research/ python research/build_report.py
```

Research notes retain narrower evidence boundaries and original source citations; they are not all
independent replications. Later review changes should be versioned without overwriting this
capsule's measurements.
