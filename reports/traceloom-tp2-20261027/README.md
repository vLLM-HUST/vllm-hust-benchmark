# TraceLoom TP2 structural comparison, 2026-10-27

This report archives a new HOST/AIV comparison on two Ascend 910B2 devices. Four engine processes
completed in HOST → AIV → AIV → HOST order. Each process produced two worker profiles. The campaign
replaces unavailable historical inputs for new analysis; it does not recover or reproduce the old
FFTS+ campaign. Current CANN rejects the historical `FFTS+` environment value.

## Experiment

The source-pinned runtime uses stock vLLM with vLLM-Ascend, Qwen3-0.6B FP16, tensor parallelism 2, a
synchronous scheduler, sequence limit 1, token budget 32, maximum model length 128 and effective
PIECEWISE graph capture size `[1]`. Prefix caching is disabled. Two sequential requests contain
eight copies of token 1000 or 1001 and generate 32 greedy tokens with EOS ignored. Each process
warms up those requests before profiling them. Output equality is checked within and across
processes. The runtime manifest, model config, hardware inventory, exact arguments and frozen
scripts travel with the evidence.

Two ranks from one engine process are paired observations, not independent trials. Replay instances
are nested within processes. This is a native-profile structural study, not request-linked
production tracing, StriaTrace execution, an HTTP service benchmark, or a formal leaderboard
submission. No unprofiled backend speedup is claimed.

## Result

All four processes complete with identical output tokens. All eight worker profiles pass the
raw-event comparison: 496 replay instances, 14,384 graph launch coordinates and 273,544 member
occurrences match. Every per-replay cost vector also matches, including task sum, interval union and
enclosing span. [summary.json](summary.json) contains the process/rank breakdown and hashes of the
complete audits.

Each replay has 295 shared members, 456 HOST-specific members and 57 AIV-specific members in both
order-balanced repetitions. Structural membership is repeatable, while the sign of the profiled span
difference changes between repetitions. The four HOST/AIV rank pairs have median spans of
20.908/19.274, 20.823/19.118, 18.063/19.768 and 17.906/19.833 milliseconds. These are descriptive
profile measurements, not an unprofiled performance ranking. The independent raw workflow returns
the same structural and cost answers; this campaign validates recovery and archives reproducible
inputs, not a superiority claim over an equally capable raw workflow.

## Data custody

The
[published evidence release](https://github.com/vLLM-HUST/vllm-hust-benchmark/releases/tag/evidence-traceloom-tp2-20261027-v1)
contains five archives, a member manifest and archive checksums. All seven assets were downloaded
from public links without authentication. Their sizes and SHA-256 digests match, and every archived
member passed streamed verification.

Each cell archive contains its original profiler files, launch and application receipts, frozen
capture scripts, offline profiler exports, both `worker-*.db` analysis databases, independent
raw-event audits and materialization logs. A separate support archive contains the complete
reproduction scripts, runtime fingerprints, model configuration, the exact analyzer binary and its
recorded patch provenance. Model weights, compiler caches and Python bytecode are excluded.
`MANIFEST.json` enumerates every archived member with its size and SHA-256; `SHA256SUMS` covers the
archives and manifest. The packager streams and hashes every member without extraction.

`release.json` and `public-verification.json` record the final asset IDs, URLs, sizes, digests and
actual download verification. Git stores the report, compact results and reproduction tools; Release
assets store the large raw profiles and eight queryable analysis databases. The evidence tag is a
storage anchor, not the measured runtime revision.

The verification receipt hashes the executed source preserved in
`verify_release.executed.py.source`. The Python tools and protocol in this report are
repository-formatted copies; the archives retain the exact inputs and sources used for capture,
analysis and packaging. Formatting does not replace those originals.

## Reproduction

After downloading the assets named in `release.json`, run `sha256sum -c SHA256SUMS`. Verify each
archive member against `MANIFEST.json` with
`reproduction/striatrace-comparison/package_tp2_evidence.py`'s streaming verifier before extracting
into a new directory. The archives use `campaign/`, `analysis/` and `support/` prefixes, so the five
assets reconstruct one capsule. The archived SQLite databases can be queried directly without
loading a model or initializing a device. Absolute source paths inside receipts are original
provenance, not download locations.

To recollect on a compatible runtime, use `run_tp2_formal.py` with explicit `--runtime`, `--model`
and a fresh `--output`; it requires the existing `npu-share` service and two-device `profiling`
ownership. It does not install or change an engine. Then run
`analyze_tp2_formal.py CAMPAIGN --output NEW_DIR --analyzer BINARY` using a Python runtime with
torch-npu's offline exporter. That workflow reconstructs all eight databases, independently checks
members and coordinates, and compares matched-region costs. Read the provenance before executing the
optional archived Linux analyzer binary.

The original raw workflow uses provider launch/wait/completion controls and captured stream
metadata. Replayed task IDs can have multiple HCCL occurrence names; the comparator preserves all
candidates and does not use an ambiguous label to establish a shared lane. Correspondence uses
complete ordered identity/kind sequences within independently checked outer coordinates.
Backend-specific lanes are retained separately rather than forced into a one-to-one operation
mapping.
