# Harbor local model transport diagnostic

The frozen Harbor 0.24.0 / Terminus-2 candidate can call this host's real Qwen3.5-35B-A3B API and
parse its returned JSON. This closes the local client/endpoint compatibility question only. It is
not a Terminal-Bench task, sandbox, verifier validation, remote-backend connectivity check or
quality score.

The diagnostic uses the [candidate job](../pyramidkv-terminal-harbor-contract-20261009/job-b0.json)
without changing its model or generation settings. The input filename's B0 label does not disable
compression: the live server remains in the original compression-enabled arm. The synthetic prompt
uses 55 input tokens and returns 23 tokens with zero cached input tokens, so it does not exercise
the 4,096-token compression threshold.

Exactly one successful query timing and one HTTP 200 chat completion were observed. The real
Terminus parser returned no error/warning and an empty command list. Its `task_complete` field is
part of an explicitly requested synthetic JSON payload, not evidence of a benchmark task being
solved. No environment was set up and no shell command or verifier was executed.

The transport started after the four-arm performance series had completed; timestamps are in
`boundary.json`. The full configuration, source and dependency identity remain in the candidate
report. This diagnostic uses its ARM64 local schema environment and does not qualify an AMD64
execution image. The earlier preparation's zero-model-call receipt remains unchanged; this separate
follow-up contains one real local query.

`receipt.json` preserves the prompt, response, usage, parser result, timings and job hash. Gzipped
files preserve the exact diagnostic and log; `server-access.log` is the matching access-log line.
The client supplies a synthetic placeholder for the locally unauthenticated endpoint. It is not an
external service credential; remote connectivity and authentication remain unconfigured.
Custom-model zero cost values are placeholders and not a monetary-cost claim.

The remaining gates are an AMD64 Docker/Harbor backend with enforced resources, a reachable model
endpoint from that backend, a frozen execution environment, fresh-sandbox verifier/reference/failure
validation, and canonical task/rights review. No official readiness, leaderboard or optimization
claim changes.
