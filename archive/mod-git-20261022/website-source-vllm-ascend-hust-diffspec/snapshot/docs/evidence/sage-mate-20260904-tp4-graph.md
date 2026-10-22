# Sage Mate Qwen3.8-27B TP4 graph qualification — 2026-09-04

## Verdict

DiffSpec is **functionally compatible, performance degraded** for the exact
Sage Mate TP4 graph lane below. It is not a speedup recommendation. Installed,
configured, enabled, and runtime effective remain separate states; runtime
effective requires the image, commits, launch contract, four-rank log markers,
and speculative-token counters recorded here.

## Immutable candidate

- vLLM-HUST `762f85b311fbab0bcf8921dd216f5093cd58b9b8`
  (`0.28.1rc1.dev319`)
- vLLM-Ascend-HUST `4e57439e58ed3d78e675f9fd7b4614fb183c5394`
  (`0.25.1rc1`)
- DiffSpec `c78f55c7e4923da342f2fc52c2cb509c150e5363`
- Runtime image `sha256:6dec9e68eaa61d5a3297abc5006d939d5644aa203c16ef1f9af65fb54d60722b`
- Wheel SHA256 `2028172d18ac978fcfdb78e7192ec794641a517222a95a3eba888175b3d6aeba`
- Target `/data/shared_models/Qwen/Qwen3.8-27B`, BF16, TP4, PP1
- Draft `VirVen/Qwen3.5-27B-EAGLE3-v2`, BF16, TP4, one layer,
  vocabulary 248,320; checkpoint SHA256
  `a57cefc45874197a24dd2a092cfd0d0f7d6a2f2cca156d09f2d2f4a56dc4e5be`
- `FULL_DECODE_ONLY`, capture sizes `[4,8,12,16]`, speculative depth 3;
  eager, async scheduling, and prefix caching disabled

## Functional evidence

- All four TP ranks loaded the draft and logged `ACLGraphWrapper`; graph capture
  completed 4/4. The final log slice contains no unexpected traceback or
  runtime error.
- Two deterministic 10-request suites returned HTTP 200 and correct answers.
  Nine of ten outputs matched the target-only text exactly; the remaining sky
  explanation was semantically equivalent.
- Four simultaneous requests returned `143`, `391`, `Au`, and `分布式系统`.
- Closing a streaming response after its first chunk drained running requests
  to zero in 0.529 seconds.
- A bad model request returned 404; the following request returned exactly
  `DIFFSPEC_R007_RECOVERY_OK`.
- A 5,425-token prompt returned exactly `DIFFSPEC_R007_LONG_OK`.
- The final contract suite passed 26 tests. The complete device matrix exercised
  sampler verification, accept/reject accounting, compact KV metadata,
  cancellation, exception recovery, concurrency, and graph capture/replay.
- Counter delta: 534 draft tokens and 103 accepted tokens, 19.29% accepted.

## Performance evidence

After graph warm-up, DiffSpec measured TTFT P50/P95 0.459/0.469 s, request
latency P50/P95 0.744/3.990 s, and output throughput P50/P95 14.00/14.24 tok/s.
The target-only baseline measured 0.314/0.333 s TTFT, 0.400/1.166 s request
latency, and 47.72/56.97 tok/s output throughput. This checkpoint and depth are
therefore functional but materially slower. Do not label the lane accelerated,
recommended, or production-ready on the basis of this run.

Raw evidence is retained under the qualification custody root
`sage-mate-mod-compat-20260904T053524Z/diffspec-r007`.

## Rollback

Disable the DiffSpec bundle and restore the exact baseline image and original
managed launch environment. A successful rollback must verify baseline image
identity, `/health`, NPU0–3 ownership, and an exact post-rollback inference.
