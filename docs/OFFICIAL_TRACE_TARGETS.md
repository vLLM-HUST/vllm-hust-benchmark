# Official production-trace workload targets

BurstGPT and TraceLab form the dedicated `production-trace` official target profile. This profile is
separate from the Qwen2.5 `core-text` targets and uses DeepSeek-R1-Distill-Qwen-32B, BF16, two Ascend 910B2
chips, tensor parallel size 2, and a 131072-token context window. Measured entries still require three
successful independent starts and an evidence-backed attestation before publication.

The baseline runtime is the official vLLM-Ascend `v0.22.1rc1-openeuler` image pinned by immutable
digest (`quay.io/ascend/vllm-ascend@sha256:bfc46fa57aedf933e6d6d4adcf42ce96aed956689018faf111bb01571891e092`).
The runner verifies the declared image reference, the pinned Python package set, both source commits,
and the local model artifact before a trace server can start. The target explicitly enables the
image's installed batch-invariant operator path with `VLLM_BATCH_INVARIANT=1`; the value is recorded
in the same-spec contract, startup evidence, and attestation. A locally derived Dockerfile is not part
of the target contract.

The immutable image does not provide the `aclnnAddRmsNormBias` operator used by the optional
`norm_quant` fusion pass. The public contract therefore records
`ascend_compilation_config.fuse_norm_quant=false` as an explicit server parameter. Both this setting
and the batch-invariant mode are part of the same-spec evidence and generated registry, not
unreported runtime workarounds.

The image's batch-invariant fused-attention operator does not expose the `.out` overload required by
FULL cudagraph capture. The contract therefore fixes `compilation_config.cudagraph_mode=PIECEWISE`:
torch compilation remains enabled, while the unsupported FULL capture path is excluded explicitly.

## Registered targets

### `burstgpt-v2-production-replay`

- Source: [HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT), release `v2.0`, asset
  `BurstGPT_3.csv`.
- License: CC BY 4.0.
- Integrity: 231,682,327 bytes, SHA256
  `2299986a07388aa303ec2c41d1131e756db650a39ed6ef9dfe7cc3d7f9a43b8f`.
- Preserved fields: arrival timestamp, session ID, input/output token counts, source model, and
  end-to-end response time.
- Replay caveat: the public data does not contain prompt text or exact cross-turn prefix overlap.
  Requests therefore use deterministic synthetic token IDs. Rows with zero response tokens remain
  part of source provenance but are counted and skipped as non-replayable source failures.
- Official replay cohort: the first 1000 replayable requests in global arrival order; overflow
  policy is fail-closed because this cohort fits the 131072-token target window. The selected cohort
  contains 713,787 input tokens and 98,745 requested output tokens. Input length ranges from 8 to
  25,085 tokens; output length ranges from 2 to 1,101 tokens.

### `tracelab-v0.0.1-coding-agent-replay`

- Source: [uw-syfi/TraceLab](https://github.com/uw-syfi/TraceLab), release `v0.0.1`, asset
  `syfi_coding_trace.jsonl.gz`.
- License: CC BY 4.0.
- Integrity: 53,601,226 bytes, SHA256
  `9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b`.
- Preserved fields: invocation timing, pseudonymous session, provider/model, prefix/append/output
  tokens, and whether the step was triggered by a user message or tool result.
- Replay caveat: the sanitized release intentionally excludes prompt, response, and tool-result
  text. Replay maintains a deterministic per-session token tape and reconstructs each request from
  the published prefix and append accounting. This preserves session evolution and accounting
  boundaries, but cannot reproduce the provider's exact token identity, semantic content, or cache
  hit ratio.
- Official replay cohort: records are globally sorted by arrival time before selecting the first
  1000 replayable requests. The current cohort has 22 sessions, 38,374,841 input tokens, 439,079
  requested output tokens, 64 user-message triggers, and 936 tool-result triggers. Over-window rows
  are excluded, counted, and retained in plan provenance; they are never silently truncated. The
  selected input lengths range from 2,645 to 120,566 tokens and requested outputs range from 1 to
  19,602 tokens.

Both official replays use an explicitly transformed arrival process: inter-arrival times are
accelerated by 60× and each transformed gap is capped at one second. The leaderboard records this
transform, the full token-length distributions, the selected-request digest, and the cohort setting
signature. These runs therefore test the same selected production cohort under a reproducible stress
schedule; they must not be described as preserving wall-clock arrival intervals.

## Usage

List or inspect the pinned contracts:

```bash
official-trace-targets list
official-trace-targets show tracelab-v0.0.1-coding-agent-replay
```

Download and verify an asset:

```bash
official-trace-targets fetch burstgpt-v2-production-replay \
  --cache-dir /data/vllm-hust-benchmark/traces
```

Preflight a replay without contacting a server:

```bash
official-trace-targets replay tracelab-v0.0.1-coding-agent-replay \
  --cache-dir /data/vllm-hust-benchmark/traces \
  --model /path/to/model \
  --max-model-len 131072 \
  --max-requests 1000 \
  --time-scale 60 \
  --max-interarrival-s 1 \
  --dry-run
```

Run against an OpenAI-compatible completion endpoint and retain provenance/results:

```bash
official-trace-targets replay burstgpt-v2-production-replay \
  --cache-dir /data/vllm-hust-benchmark/traces \
  --model /path/to/model \
  --base-url http://127.0.0.1:8000 \
  --max-model-len 131072 \
  --max-requests 1000 \
  --max-concurrency 64 \
  --time-scale 60 \
  --max-interarrival-s 1 \
  --output results/burstgpt-v2-replay.jsonl
```

Context overflow fails closed by default. `--overflow-policy truncate-input` is available only for
explicitly labelled diagnostic runs; truncation changes the workload contract and must not be mixed
with exact-trace results. Likewise, `--time-scale` and `--max-interarrival-s` are recorded in output
metadata because they change the arrival process.

## Attribution

- Yuxin Wang et al., “BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems,” KDD
  2025, <https://doi.org/10.1145/3711896.3737413>.
- Kan Zhu et al., “TraceLab: Characterizing Coding Agent Workloads for LLM Serving,” 2026,
  <https://arxiv.org/abs/2606.30560>.
