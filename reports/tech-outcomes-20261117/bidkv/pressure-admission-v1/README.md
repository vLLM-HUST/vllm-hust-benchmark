# BidKV bounded pressure admission, zero generation only

This is a separate supplemental mechanism diagnostic. It never replaces the full frozen five-dataset
MOD cohort. The primary v4 source packet stays unchanged. This first packet contains no pressure
inference client and submits no `/v1/chat/completions` traffic.

B0 and B1 retain the same prepared vLLM0.23 scheduler `ae12773b…` and copied installed Ascend
package with CPU parser `b7a84571…`. CPU binding stays True, with the same model/SDK, TP2, PP1, DP1,
MTP2, APC, async scheduling, FULL_AND_PIECEWISE graph, seq16, batch4096, maxlen65536 and image1. The
separate shared memory-pressure setting is exactly1GiB KV per rank. Fresh namespace/cache/temp paths
preserve prior attempts.

The two additional shared observation settings are
`--worker-extension-cls bidkv_probe_cache_evidence.BidkvProbeCacheEvidenceWorkerExtension` and
`VLLM_SERVER_DEV_MODE=1`. They add an official metadata-only worker method and the official
localhost dev RPC router. No core/Ascend method or optimization is patched or disabled. The server
stays on127.0.0.1. Its dev router is a disclosed supplemental-profile difference from the primary
cohort.

The single allowed RPC is `POST /collective_rpc` with
`{"method":"describe_bidkv_probe_cache_groups_v1","args":[],"kwargs":{},"timeout":30}`. It returns
`{"results":[rank0_metadata,rank1_metadata]}`. The extension reads the actual initialized runner
cache configuration: group type/layer names/count, block/page sizes, dtype/shape metadata,
speculative-state block counts, K and Python KVCacheTensor size/shared_by descriptors. It never
reads actual tensor values, repr, storage addresses, copies or dumps; it never modifies a live
config or allocator. It also reports exact loaded-module/source identities and actual shared
optimization settings. CPU poison guards check these boundaries.

Official interfaces are pinned to vLLM commit `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`:

- [Worker extension loader](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/worker/worker_base.py)
  checks attribute conflicts before composing the worker class.
- [CLI argument registration](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/engine/arg_utils.py)
  registers `--worker-extension-cls` (actual installed line1113).
- [API server](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/entrypoints/openai/api_server.py)
  gates the dev routers on `VLLM_SERVER_DEV_MODE`.
- [RPC handler](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/entrypoints/serve/dev/rpc/api_router.py)
  accepts a method string, string argument containers and timeout, returning native JSON metadata.
- [Ascend initialized config](https://github.com/vllm-project/vllm-ascend/blob/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd/vllm_ascend/worker/model_runner_v1.py)
  stores the runtime `self.kv_cache_config` during initialization (actual installed line3849).

`official-source-identities.json` records each actual installed file's SHA256/Git blob and confirms
its tracked bytes match the pinned commit. A source pin or CPU guard is not actual NPU admission.

Two gates remain mandatory. First, the parent independently publishes/reads back this
zero-generation packet. Only after the complete primary judge/quality/controller finishes with real
exits and both allocated devices are released may `start_native_admission.py` cold-load Native. It
creates one non-resettable60min clock, including both future cold initializations and both
publication parks. Its supervisor actually waits for the model owner; the owner records API wait
codes and release observations. A host restart invalidates the clock instead of extending it.

Second, Native must become actually healthy, report both ranks'1GiB cache-group units and actual
unchanged optimization settings, and match the original eight16407-token input hashes through
`/tokenize`. Then the fixed capacity/admission/growth calculation and go/no-go decision must be
published and independently read back with the final pressure-client source/plan. No inference
starts from this package alone. A failed model admission or a fixed workload that cannot create
pressure is retained as zero generation; parameters are not changed on site.

The proposed final inference workload is fixed before any probe outcome: the original eight16K C8
inputs,8lanes,3batches/arm,48requests,4096accepted output tokens/request with ignore_eos=True,
maximum196608output tokens. Every request will have a unique fresh cache salt. This is synthetic
mechanism evidence, never full-five EOS/quality evidence. Prior256/1024proposals remain unexecuted.
The existing16GiB cache metadata showed2048-token blocks, explaining why1024decode tokens might not
grow an input allocation; actual1GiB K/group units must still be observed, not inferred
from730blocks or a linear token projection.

All rounds, calls/selections/liveness/fallback/failure counters, initialization/compile/park costs,
negative values, zero-call outcomes, deadline partial captures and unexecuted requests remain in
custody. No adaptive KV/output/workload choice or best-round selection is permitted. The60min
generation/operation deadline has separately reported owned-cleanup grace; it is not extended to
finish missing requests. Once the bounded diagnostic releases the devices, another project may reuse
the machine after its own admission.
