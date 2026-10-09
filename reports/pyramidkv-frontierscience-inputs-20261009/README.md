# FrontierScience input contract for PyramidKV

This follow-up advances
[PyramidKV #8](https://github.com/vLLM-HUST/vllm-ascend-pyramidkv-hust/issues/8) and
[benchmark #254](https://github.com/vLLM-HUST/vllm-hust-benchmark/issues/254). It freezes input
manifests for both tracks, with no solver or judge calls and no score. It does not change the
earlier MMLU-Pro evidence or the official v0.18 FP16 leaderboard.

## Verified inputs

The exact Hugging Face `openai/frontierscience` revision is
`25ed67db7da8f4591484e764008ff585544f5a30`. Both JSONL files were downloaded locally and checked
against the SHA-256 values recorded in `summary.json`; the pinned dataset card declares Apache-2.0.
The local retrieval covers these two files, not the separate host's complete 1,135-file asset audit.

| Track    | Rows | Unique task_group_id | Rendered prompt tokens | Above 4,096 |
| -------- | ---: | -------------------: | ---------------------: | ----------: |
| Olympiad |  100 |                  100 |                 94–835 |           0 |
| Research |   60 |                   59 |              164–1,660 |           0 |

**Research group IDs are not unique.** The immutable manifest uses track plus zero-based source row
as its task ID and retains the group ID and canonical row hash. All 60 Research rows remain present;
deduplicating by group ID would silently remove a task. The manifests publish hashes and lengths,
not source questions, reference answers or rubrics.

The candidate prompt is one user message containing only `problem`, rendered by the pinned Qwen chat
template with thinking disabled and `add_generation_prompt=True`. The source `answer` field is never
supplied to the solver. The tokenizer files, model revision and every rendered prompt hash are
recorded. This is an input-only candidate, not a claim to reproduce the published evaluation. No
candidate prompt crosses the prefill threshold, so this framing is not naturally eligible for
PyramidKV prefill compression. This does not predict full agent trajectories, a different prompt
contract or task accuracy; any such change requires a new length audit. Do not pad prompts or lower
the compression threshold to manufacture applicability.

## Remaining grading contract

[grading-contract.json](grading-contract.json) records separate track requirements and leaves
unselected execution fields explicitly null. The
[official description](https://openai.com/index/frontierscience/) uses a model-based judge for
short-answer equivalence in Olympiad and rubric-based scoring in Research; Research is correct at
7/10 or higher. Therefore, a generic string exact-match scorer cannot stand in for both tracks. The
source `answer` is a short answer in Olympiad and grading material in Research, and must remain
outside solver prompts.

Judge revision, prompt, parser, calibration, runtime image, trial counts, generation/tool budget,
failure policy and per-track aggregation still need a frozen, reviewed execution contract. No
third-party harness has been silently adopted as an official scorer. Status remains **blocked**;
this report supplies input preparation, not task-quality acceptance or an optimization claim.

## Reproduction

Download `README.md`, `olympiad/test.jsonl` and `research/test.jsonl` from the pinned dataset
revision into an asset directory. Use the tokenizer from the model revision recorded in
`summary.json` and the prior experimental BF16 environment (Transformers 5.14.1). Run:

```bash
python scripts/audit_pyramidkv_frontierscience.py \
  --assets /path/to/frontierscience \
  --model /path/to/Qwen3.5-35B-A3B \
  --model-revision 59d61f3ce65a6d9863b86d2e96597125219dc754 \
  --output /path/to/new-empty-output-directory
```

The script rejects changed source bytes and existing output directories. Compare both generated
inventories and `summary.json` byte-for-byte with this report; `SHA256SUMS` covers the published
report and script snapshot. Source questions and answers stay in the local downloaded assets.
