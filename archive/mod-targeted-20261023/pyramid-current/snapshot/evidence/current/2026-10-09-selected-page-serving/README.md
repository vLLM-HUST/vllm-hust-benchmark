# Original vs selected-page serving diagnostic

This compares original method revision `47580977a2dca108054470b3132811bd690020a6` with candidate
`508a4eadb5e1b3f148975d362e13bf78e688b6da`. Both arms enable PyramidKV with the same qualified
Qwen3.5-35B-A3B BF16 TP2 graph/MTP2 profile. It is not compression-off B0 versus compression-on B1.

Each arm completed 409 requests with zero failures. All 300 quality outputs and per-task F1 scores
were identical: NarrativeQA 28.593067, Qasper 50.315085, and 2WikiMQA 63.9. The full, unchanged
request plan is referenced in the frozen contract; the raw outputs and per-request hashes are
included here. These are named sampled workloads, not full-dataset scores.

There are **33 synthetic performance output differences**: 32 at concurrency four and one at
concurrency one (`perf-24576-c1-r0-00`). Seven differences are on 1,024-token requests, below the
compression threshold. The cause is not established by this comparison. Five additional candidate
repetitions of the disputed C1 request, with distinct cache salts and zero cached prompt tokens,
produced identical outputs to one another and to the candidate arm. This does not yet establish
baseline repeatability or explain the baseline/candidate difference. An earlier attempt to reset
prefix caching returned HTTP 404; its failure log is retained.

**No serving performance improvement is claimed.** CPU-heavy repository tests and pre-commit checks
overlapped the original arm. Its timings remain as raw diagnostic observations, but are unsuitable
for a causal speed comparison. Dedicated matched performance arms must be rerun without competing
test, lint, tokenizer or NPU diagnostic work. Earlier published negative end-to-end compression
results remain unchanged.

The archive includes SSE streams, per-request results, metrics/telemetry, arm summaries, the frozen
contract, runners and repeatability records. Concatenate `raw.tar.gz.part-*` in filename order and
verify the combined SHA-256 against `archive.json` before extraction. `members.json.gz` lists each
member's hash. Filenames and workload identifiers retain their original run identities.

Real-cache shadow checks are a separate diagnostic: both methods operate on the same live query and
source cache, with original destination pages restored between them. Those checks are not timing
measurements and cannot alone prove performance or every possible serving configuration.

The live-cache check completed **20 requests, 40 worker comparisons and 880 K/V tensor
comparisons**, all successful and bitwise equal. Every request paired with both TP workers; 20
scheduler commits and 40 worker acknowledgements are present. It includes 8K, 24K and 32,704-token
inputs, concurrency four, three quality tasks and an APC repeat reporting 20,480 cached prompt
tokens. The CPU self-test deliberately injects a changed cache value and confirms rejection; that
expected failure is separate from the NPU success counts.

`shadow-raw.tar.gz` contains the exact overlay, baseline reference, diagnostic scripts, worker
records, requests, deployment command and server log. `shadow-members.json.gz` gives member hashes.
The overlay is specific to this isolated runtime; reproduction requires adjusting its local paths
and placing `shadow-src` on `PYTHONPATH` before a fresh managed launch. It changes only the owned
PyramidKV method for the diagnostic, leaving the shared host classes untouched. The production
candidate source itself is unchanged. The public shadow plan includes task/prompt hashes and
budgets; use the original frozen evaluation plan to reconstruct the prompt text.
