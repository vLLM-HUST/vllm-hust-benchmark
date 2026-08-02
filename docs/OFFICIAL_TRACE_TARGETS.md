# Official production-trace workload targets

BurstGPT and TraceLab are official **workload targets**, not active public leaderboard baselines.
They remain `provisional` until matched vLLM/vLLM-HUST runs establish a model, hardware, server
configuration, and repetition contract. This boundary prevents trace metadata from being presented
as measured 910B2 performance.

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

### `tracelab-v0.0.1-coding-agent-replay`

- Source: [uw-syfi/TraceLab](https://github.com/uw-syfi/TraceLab), release `v0.0.1`, asset
  `syfi_coding_trace.jsonl.gz`.
- License: CC BY 4.0.
- Integrity: 53,601,226 bytes, SHA256
  `9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b`.
- Preserved fields: invocation timing, pseudonymous session, provider/model, prefix/append/output
  tokens, and whether the step was triggered by a user message or tool result.
- Replay caveat: the sanitized release intentionally excludes prompt, response, and tool-result
  text. Synthetic payloads keep the reported prefix stable per session and make the reported append
  request-specific. This preserves the accounting boundary but cannot reproduce the provider's
  exact token identity, semantic content, or cache hit ratio.

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
  --max-model-len 32768 \
  --max-requests 10000 \
  --time-scale 10 \
  --output results/burstgpt-v2-replay.jsonl
```

Context overflow fails closed by default. `--overflow-policy truncate-input` is available only for
explicitly labelled diagnostic runs; truncation changes the workload contract and must not be mixed
with exact-trace results. Likewise, `--time-scale` and `--max-interarrival-s` are recorded in output
metadata because they change the arrival process.

## Attribution

- Yuxin Wang et al., “BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems,”
  KDD 2025, <https://doi.org/10.1145/3711896.3737413>.
- Kan Zhu et al., “TraceLab: Characterizing Coding Agent Workloads for LLM Serving,” 2026,
  <https://arxiv.org/abs/2606.30560>.
