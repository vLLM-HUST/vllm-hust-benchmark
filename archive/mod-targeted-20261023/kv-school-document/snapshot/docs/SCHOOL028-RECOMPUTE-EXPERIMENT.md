# School 0.28 bounded recompute experiment

This opt-in adapter is an experimental compatibility path, not support for the
segmented materialization host API. It requires the school build
`0.28.1.post1.dev143+gf18cf803c` and exact SHA256 pins for the chat serving and
engine request modules. Other source revisions fail before hooks are installed.
Default plugin registration and the supported native host remain unchanged.

Enable `kv_materialization` in `VLLM_PLUGINS` and set
`VLLM_KV_ENABLE_SCHOOL_028=1` explicitly. Install the committed plugin with
`--no-deps` into an isolated target; do not replace the serving runtime. Record
the plugin commit, actual import path, host versions, source hashes, policy log
and runtime event log with each experiment.

Only text token prompts and effective `recompute` decisions are accepted. The
existing controller supplies a request-scoped cache salt. Token IDs, decoding
parameters and model weights are unchanged. Non-recompute decisions and
multimodal prompts fail closed. Requests without reuse anchors are not given
synthetic anchors or shared-prefix hints to improve benchmark results.

The `school028_request_salt_admitted` event proves the salt reached the engine
request. It does not prove cache reuse, avoided computation or performance
improvement. This experiment does not exercise partial reuse or transfer.
Same-workload measurements and correctness checks are required before selecting
any result as B1. The adapter alone is not qualification evidence.

## Measured MMLU-Pro screen

The 2026-10-19 school-host screen is archived in the
[benchmark midterm report](https://github.com/vLLM-HUST/vllm-hust-benchmark/blob/84f0443be92c658bb658b281fcb4241eb5a9fb97/reports/vllm-hust-midterm-20261019/README.md).
Its `b1-kvmat-screen-evidence.tar.gz` contains the raw requests, policy and engine
receipts, source pins, device samples and clean exit/release evidence. The
`b1-kvmat-screen.json` summary is verified from that archive.

The measured installed Python sources exactly match commit
`b1680f2365f3468a062a0530bd50dba3d8f80046`. The later CI correction changes the
unsupported-prompt exception from `ValueError` to `TypeError`; it does not alter
the measured text/recompute path. The running isolated payload was not edited.
The native Extension Manager manifest remains incompatible with this 0.28 host;
the measurement explicitly opted into the bounded adapter instead.

All 200 measured requests completed. Policy and engine cache-salt digests match
for all 200 requests and 282,265 input tokens. All decisions were recompute;
no reuse anchors were added and partial reuse was not exercised. The result is
a single-run service-performance screen, not a task-accuracy or statistical
significance claim.
