# Issue #3: long-context TTFT admission, pre-submission diagnostic

This is a **pre-submission engineering diagnostic**, not a formal task-quality
benchmark. The experiments used Qwen3.5-35B-A3B W8A8, one Ascend 910B3 card
(physical card 4), TP1, eager execution, native APC off, chunked prefill off,
block size 128, `max-num-seqs=4`, temperature 0, thinking disabled, and
sequential requests. The answer cap was 64 tokens; all answers finished in 2–3
tokens. Each row below compares the same ordered inputs on the same card.

| Input band    | Plugin configuration                   | Correct short answers | Semantic hits | Target TTFT median | 12-request throughput |
| ------------- | -------------------------------------- | --------------------: | ------------: | -----------------: | --------------------: |
| 2.2k–4.1k     | Baseline, plugin disabled              |                 12/12 |             0 |           0.5161 s |          1.0958 req/s |
| 2.2k–4.1k     | Existing `tail_anchor`, 512-token tail |                  5/12 |             8 |           0.5342 s |          0.9891 req/s |
| 2.2k–4.1k     | Existing `tail_anchor`, 256-token tail |                  5/12 |             8 |           0.5177 s |          1.0232 req/s |
| 2.2k–4.1k     | Existing `tail_anchor`, 128-token tail |                  5/12 |             8 |           0.5217 s |          1.0124 req/s |
| 2.2k–4.1k     | New wheel, `MIN_PROMPT_TOKENS=8192`    |                 12/12 |             0 |           0.5307 s |          1.0946 req/s |
| 6.9k          | Baseline, plugin disabled              |                 12/12 |             0 |           0.5402 s |          1.0945 req/s |
| 6.9k          | Existing `tail_anchor`, 512-token tail |                  5/12 |             8 |           0.5366 s |          0.9580 req/s |
| 16k, repeat 1 | Baseline, plugin disabled              |                 12/12 |             0 |           1.0191 s |          0.7245 req/s |
| 16k, repeat 1 | Existing `tail_anchor`, 512-token tail |                  5/12 |             8 |           0.5609 s |          0.8009 req/s |
| 16k, repeat 2 | Baseline, plugin disabled              |                 12/12 |             0 |           1.0189 s |          0.7199 req/s |
| 16k, repeat 2 | New wheel, `MIN_PROMPT_TOKENS=8192`    |                  5/12 |             8 |           0.5512 s |          0.8109 req/s |

The new wheel's 16k repeat reduced median TTFT on the eight reuse targets by
**45.9%** and increased full 12-request throughput by **12.6%** relative to
its paired baseline. The first 16k repeat with the existing wheel gave a
similar **45.0%** median TTFT reduction. The new wheel logged 8 semantic hits
and 124,260 skipped prompt tokens. This is a long-context speed result for
this constructed workload, not a gain on the original 12 short prompts. The
short-prompt admission run was not adjacent to its baseline, so its small
TTFT difference is not a robust paired speed estimate.

**Quality cost:** seven of eight edited hit requests returned the cached
source's old answer in both 16k runs. The baseline scored 12/12; the
approximate path scored 5/12. The new admission floor prevents these hits for
the original shorter prompts and restores their 12/12 outputs, but it does not
repair approximate reuse above the floor. Do not enable this configuration for
fact-sensitive tasks requiring answer accuracy guarantees.

## What changed

`VLLM_ASCEND_SIMLLM_MIN_PROMPT_TOKENS` is a new nonnegative, optional
configuration value. It defaults to `0`, preserving the old policy. With
`8192`, prompts below the floor are excluded from both matching and KV
snapshot creation. Longer prompts follow the same `tail_anchor` path as before.
The 2.2k–4.1k run verified zero reuse and zero snapshot log events with the
new wheel, and its twelve outputs exactly matched the disabled baseline.
The 16k run verified eight real reuse events and a lower TTFT. This admission
rule targets latency by avoiding short requests where KV restoration did not
amortize. It is not a general recommendation for concurrency or throughput:
earlier 5,000-request, one-output-token replays showed benefits on shorter
prompts under a different concurrency regime, which an 8192-token floor would
exclude.

Profiling of the original 12 requests with `PROFILE=1` measured approximately
3–4 ms for matching and 13 ms for KV restoration on a hit, with about 250 ms
for the prefill forward and 225 ms for a decode forward. Profiling synchronizes
the NPU and changes timing, so use it only to locate work, not as a performance
score. Changing the tail from 512 to 256 or 128 tokens did not yield a
repeatable short-prompt TTFT gain.

## Reproduce and verify

The frozen base input is [`evidence/workload.jsonl`](evidence/workload.jsonl).
[`make_long_workload.py`](make_long_workload.py) extends each group's shared
background before the final question. With the pinned model tokenizer, it
reproduces `evidence/workload_long.jsonl` at `--target-tokens 6900` and
`evidence/workload_16k.jsonl` at `--target-tokens 16000` byte for byte. The
repeated neutral filler is synthetic and can alter model behavior; it is
reported separately from the original 12-request workload.

On the qualified `train8-docker` host, source the CANN 9.1 and ATB setup
scripts, use the original installed-wheel harness settings, and run
[`reproduce.py`](reproduce.py) with the new and rollback wheels under
`evidence/wheels/`. Its `MIN_PROMPT_TOKENS=8192` setting is
injected only for the new wheel, because the older wheel rejects unknown
configuration even when disabled. The first attempted paired run failed at
this validation stage, before model loading; its raw log is retained under
`evidence/run-guarded-long/`. The corrected paired run is
`evidence/run-guarded-long-r2/`. The raw executed harness is retained as
`evidence/run_guarded.py.gz`. Its final `INFERENCE_AND_WHEEL_ROLLBACK_PASS`
print was a legacy label and **does not mean a rollback arm ran in these
two-stage admission experiments**. The clean reproducibility script corrects
that label. A separate five-arm short experiment did verify exact rollback
output equality.

From this report directory on a qualified host, with CANN/ATB already sourced
and a checksum-verified local model copy at `$MODEL_DIR`, run:

```bash
python reproduce.py \
  --final-wheel evidence/wheels/new/vllm_ascend_simllm-0.1.0.dev0-py3-none-any.whl \
  --rollback-wheel evidence/wheels/rollback/vllm_ascend_simllm-0.1.0.dev0-py3-none-any.whl \
  --model "$MODEL_DIR" --device 4 --port 8154 \
  --input evidence/workload.jsonl --output /tmp/issue3-guarded-short
python reproduce.py \
  --final-wheel evidence/wheels/new/vllm_ascend_simllm-0.1.0.dev0-py3-none-any.whl \
  --rollback-wheel evidence/wheels/rollback/vllm_ascend_simllm-0.1.0.dev0-py3-none-any.whl \
  --model "$MODEL_DIR" --device 4 --port 8154 \
  --input evidence/workload_16k.jsonl --output /tmp/issue3-guarded-long
```

Use fresh output directories and a free card/port. The harness runs one
server at a time and records the full launch command and installed wheel
identity for each arm.

[`analyze.py`](analyze.py) recomputes [`comparisons.json`](comparisons.json)
from the raw JSONL and server logs, maps each worker cache event to a client
request ID, and checks short rollback and guarded output equality. Run it
with Python 3.11 and no NPU. `SHA256SUMS` covers all retained scripts and
receipts. Raw logs and metrics are gzip-compressed without changing their
contents. The new wheel SHA-256 is
`98997de3e3ee769803958770f29abe437e12fe435e73fe628749e05dee2a8d5b`;
the old enabled wheel is
`7ae77f00e47a4de1ad34acbbd02db57c4d0d565af2184ceee62490f621566666`.
The copied 22 model files were checked against the previously frozen model
manifest; all SHA-256 values matched. No model weights are included. The
temporary model copy was removed and all owned services stopped.

The targeted remote tests passed **74/74**. The full non-model suite passed
**248**, skipped 2, and deselected 4 model tests; the manifest integration
test was excluded because `vllm_hust_ext` is absent on that host. Initial
test-collection errors came from an incomplete temporary source copy; an
evidence test also needed a Git index in that copy. Both setup issues were
corrected before the full-suite pass. Local Ruff and diff checks passed.

## Limits

- Twelve requests per band and one physical card are diagnostic, not a stable
  population estimate. The baseline always ran before the plugin arm; two
  16k repeats provide consistency but not a reversed-order trial.
- The 16k input uses repeated synthetic filler and is not one of the five
  Pujiang-specified datasets. The outputs have a deterministic short-answer
  rubric, not a broader task-quality score.
- The 8192-token floor is an empirical admission setting between the measured
  6.9k no-gain band and 16k gain band. It is not a proven universal break-even
  threshold. Native APC was compared in the earlier issue #3 diagnostic, not
  rerun here.
