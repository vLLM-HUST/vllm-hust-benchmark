# SimLLM issue #3: targeted answer fidelity and native APC diagnostic

**Exploratory diagnostic, not a production quality rate or formal speed benchmark.** On 2026-10-30,
we replayed 12 deterministic, short-answer chat requests against Qwen3.5-35B-A3B W8A8 on the same
Ascend 910B3 physical card 7, TP1, eager mode. Four groups changed a number, negation, subject or
instruction inside a long shared background. Each group had one source and two edited requests. Each
generated answer was complete (2 or 3 output tokens; cap 64). The ordered input was identical across
arms. Source and target prompts shared an initial block before the changed field, then a long common
suffix. The benchmark is deliberately adversarial to approximate KV reuse.

| Arm                       | Plugin   | Native APC | Chunked prefill | Correct short answers | Same text as baseline | Median TTFT on reuse targets |
| ------------------------- | -------- | ---------- | --------------- | --------------------: | --------------------: | ---------------------------: |
| Baseline                  | disabled | off        | off             |                 12/12 |                 12/12 |                     0.5262 s |
| Approximate `tail_anchor` | enabled  | off        | off             |                  5/12 |                  5/12 |                     0.5337 s |
| Chunked baseline          | disabled | off        | on              |                 12/12 |                 12/12 |                     0.7918 s |
| Native APC                | disabled | on         | on              |                 12/12 |                 12/12 |                     0.8056 s |
| Rollback                  | disabled | off        | off             |                 12/12 |                 12/12 |                     0.5390 s |

The approximate arm logged **8/12 semantic hits**, skipping 22,556 prompt tokens. Seven of eight
edited hit requests returned the cached source's old answer instead of the edited answer; the other
hit happened to retain the same correct answer after a negation change back to the source value. All
four cold requests were correct. [Per-request comparisons](comparisons.json) include every full
output, expected answer, cache source ID, skipped-token count, cosine and TTFT. The source IDs map
to the first request in each group. **No reuse target became a later cache source** in this default
`tail_anchor` run; source rollover was therefore not exercised. The current implementation only
promotes such targets under its `aggressive` hybrid policy.

Native APC returned the same 12 answers as its chunked baseline. Its server counter recorded 2,048
prefix-cache hit tokens out of 39,794 queried tokens (about 5.1% by that counter), not a per-request
hit rate. The host rejected APC with chunked prefill off because the Qwen3.5 Mamba cache `align`
mode requires chunked prefill. We retained that
[failed startup log](evidence/run1/failed_native_apc_no_chunked/server.log.gz), then ran both
chunked baseline and APC with chunking on. This is why APC timing is compared with the chunked
baseline rather than the no-chunking baseline.

The 12-request sequential run showed **no measured TTFT gain for approximate reuse**: median TTFT on
the eight reuse targets was 0.5337 s versus 0.5262 s for baseline. These small-sample timings are
diagnostic only. They do not replace the separate 5,000-request prefill throughput replay, and the
different chunking configuration prevents treating the native APC arm as a direct speed ranking
against SimLLM. No result here establishes long-generation answer quality.

## Frozen inputs and runtime

- [Workload](workload.jsonl): SHA-256
  `eff808dd32306cf61fbf04060a62b04cfebc924d5ee60763b2a0c2fe8c3bbf1d`.
  [`make_workload.py`](make_workload.py) derives it from the
  [pinned plugin fixture](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/blob/28f2297c58cec2249576dff2ccdd251c662ad8d6/benchmarks/fixtures/qwen35_warmup30.jsonl),
  whose SHA-256 is `90f3c3229d96237b389e2a3f45fa8f8618d270921091211166661fec4a4e2c41`.
- Final enabled wheel SHA-256: `7ae77f00e47a4de1ad34acbbd02db57c4d0d565af2184ceee62490f621566666`;
  disabled rollback wheel: `10c0db015afa70b34277c8272814e5ffca0715350344041a7c6b384c4498dfea`. The
  final wheel's Python files match plugin `main` at the time of this run. The original wheels are
  retained in the
  [plugin lifecycle evidence](https://github.com/vLLM-HUST/vllm-ascend-simllm-hust/tree/b649015ec385cab3a345daf26169155a21ad0548/benchmarks/results/issue2_inference_rollback_20261019/wheels).
- Model config SHA-256: `5e4d7f74fec2f360eb9cfbfcd6ec0c4c76e684d3a11caaed259d9fd9bfbc7944`. A
  temporary local 37 GiB copy of all 22 checkpoint files was verified by SHA-256; the
  [staging manifest](evidence/model-staging.json) preserves their hashes. No model weights are
  published. The copy came from `/data/jxd/Qwen3.5-35B-A3B-W8A8`.
- Python 3.11.16; vLLM `0.28.1.post1.dev143+gf18cf803c.empty`; vLLM Ascend
  `0.25.1rc2.dev125+hust.20260903.4.g74f0c0a27`; torch `2.13.0+cpu`; torch-npu `2.13.0rc1`; CANN 9.1
  and ATB. TP1, block size 128, max model length 8192, max sequences 4, memory utilization 0.85,
  seed 0, temperature 0, chat thinking disabled, sequential concurrency 1.
- SimLLM enabled arm: `HYBRID_REUSE_MODE=tail_anchor`, suffix recomputation 512 tokens. Original
  installation hashes before and after match; stage wheels were checked against installed Python
  files. [Lifecycle summary](evidence/run1/summary.json) records activation, commands, revalidation
  and rollback equality.

## Reproduce and inspect

Generate the workload from the pinned plugin fixture:

```bash
python reports/simllm-issue3-targeted-20261030/make_workload.py \
  /path/to/pinned/plugin/benchmarks/fixtures/qwen35_warmup30.jsonl \
  reports/simllm-issue3-targeted-20261030/workload.jsonl
```

On the qualified `train8-docker` host, copy the workload, [`client.py`](client.py),
[`run_quality_apc.py`](run_quality_apc.py), [`run_on_910b3.sh`](run_on_910b3.sh) and
[`stage_model.py`](stage_model.py) to `/tmp/simllm-issue3-quality-oct30/`. Place both pinned wheels
under its `wheels/final/` and `wheels/rollback/` folders. With device 7 and port 8147 free, run
`stage_model.py`, then `run_on_910b3.sh`. The harness uses a disposable Python overlay, launches one
server at a time, captures full responses and stops each owned process. The first attempt stopped at
the incompatible APC configuration; its failed stage was moved to `failed_native_apc_no_chunked`,
and the corrected harness resumed the two completed arms with `--resume-completed`. Both attempts
and the revalidation record are retained under [evidence](evidence/). Raw `.log` and `.prom`
receipts are stored as gzip files to preserve their original whitespace and line endings through
repository format checks; decompressing them restores the exact server output.

Recompute the result without NPU access:

```bash
python reports/simllm-issue3-targeted-20261030/analyze.py \
  reports/simllm-issue3-targeted-20261030/evidence/run1 \
  reports/simllm-issue3-targeted-20261030/workload.jsonl \
  /tmp/issue3-targeted-comparisons.json
cmp /tmp/issue3-targeted-comparisons.json \
  reports/simllm-issue3-targeted-20261030/comparisons.json
```

The answer rubric extracts exactly one permitted value for each group from the entire decoded
response; ambiguous or missing values fail. Raw outputs remain available even if the rubric is
changed. This diagnostic establishes a concrete skipped-fact failure mode for the tested policy and
prompts. A broader task-quality score, long-generation experiment, source-rollover experiment and
matched performance study remain separate work.
