# StateAxis Prefix Cache Policy

Extension ID: `org.vllm-hust.stateaxis-prefix-cache-policy`

This default-off MOD lets Extension Manager own the deliberate activation of
vLLM's built-in automatic prefix cache for workloads whose repeated token
prefix reaches at least one effective host cache page. It adds no cache
algorithm and makes no novelty claim. In particular,
it is distinct from `stateaxis-automatic-prefix-cache`, the Native Engine's
lease-safe bounded-LRU research mechanism.

Status: **workload-scoped experimental positive; not performance-qualified**.

The OFF contract requires `--no-enable-prefix-caching`. When this MOD is enabled
and configured with `launch_options.enable_prefix_caching=true`, Extension
Manager removes that negative flag and appends `--enable-prefix-caching`. This
makes ON/OFF attribution explicit and keeps discovery alone inert.

On the materialized MMLU-Pro performance workload, three fresh-service,
alternating-order Qwen3-8B ON/OFF pairs showed a median output-throughput delta
of **+11.00%**, median mean-TTFT delta of **-15.39%**, and median mean-TPOT
delta of **-5.68%**. All arms completed 40/40 requests, and every ON arm
recorded 40,960 token hits from 55,102 queries (74.34%). This is a repeated,
real-online, workload-scoped experimental result—not a Qwen3.5-35B result,
accuracy score, general performance claim, or official qualification. See
[`evidence/mmlu-pro-qwen3-8b-repeated-20261015/RESULT.json`](evidence/mmlu-pro-qwen3-8b-repeated-20261015/RESULT.json).

The official Qwen3.5-35B-A3B MMLU-Pro materialization is a documented negative
boundary, not a contradiction of the 8B result. Qwen3.5 hybrid-cache `align`
mode makes the effective attention page 2,048 tokens, while no pair among the
200 official prompts shares a full 2,048-token prefix. An OFF/ON/OFF smoke
bracket therefore recorded zero local-cache hits; the apparent cold-first
speedup vanished and reversed against the warmed trailing OFF arm. A full run
cannot exercise this mechanism and was intentionally not promoted as a
performance comparison. See
[`evidence/mmlu-pro-qwen3.5-35b-boundary-20261016/RESULT.json`](evidence/mmlu-pro-qwen3.5-35b-boundary-20261016/RESULT.json).

Every result must record the workload hash, model, host commits, cache query/hit
counters, correctness, lifecycle, and resource release. Official
Before enabling this MOD, tokenize the exact workload and compare shared-prefix
lengths with the model/runtime's effective cache page. Output exactness and a
representative long-prefix Qwen3.5-35B-A3B TP2 run remain qualification gates.

Maintainer: Shuhao Zhang (Tony), directly responsible; no advisor is declared.
