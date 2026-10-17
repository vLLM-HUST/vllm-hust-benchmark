# FrontierScience independent performance

| Median across three formal runs | Native | Controller |
|---|---:|---:|
| Output tokens/s |474.1268|432.1431|
| Mean TTFT, seconds |40.9009|44.7991|
| Mean TPOT, milliseconds |29.9314|31.8879|

Controller output throughput is8.85% lower by ratio of arm medians. All six formal runs completed160/160 requests with0 failures; each produced40960 output tokens. One warmup per arm is preserved and excluded. Actual vllm measurement durations, not wrapper startup time, are used for rate calculations. Each metric is independently median-aggregated; no single run has all median values.

Same frozen160 inputs across every point; output cap256/naturalEOS, temperature0/request seed0/request-rate inf/no client cap, alternating arm order. This is separate from full quality generation and does not measure full correct-solution cost. Single-turn forwarding has no program scheduling action; the observed overhead cannot be called a throttling benefit. Quality-pair scores are lower as separately reported.

Private archive SHA in checkpoint.json; archive contains internal service references and frozen input prompts and is not public. Independent verify --archive passed raw result hashes, input hashes, command settings, totals/duration arithmetic and all formal medians. TTFT/TPOT are original vllm client aggregate statistics; per-request timing records were not retained, so their means cannot be independently reconstructed. No stronger raw-timing claim is made. Public mode verifies sanitized arithmetic and checksums, not private raw bytes.
