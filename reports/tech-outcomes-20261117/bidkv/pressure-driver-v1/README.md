# 未执行的48请求方案：已在任何生成前缩小

实际cache容量仅可首准入2请求；为保留双臂配对机会，在0生成时另冻[每臂8条v2](../pressure-driver-v2/README.md)。本目录40eb12成员保留0执行，不再授权运行，原clock不重置。以下为原方案说明。

# Supplemental BidKV pressure probe, fixed before generation

This is a separate synthetic mechanism check after the complete five-dataset cohort. A 1 GiB KV
limit per rank is artificial pressure on two 64 GiB cards. It does not represent the normal 16 GiB
profile, a task-quality score, hardware cost savings, or proof of any of the four final project
targets. All rounds, negative results, zero-call results, failures, timeouts and unexecuted cells
remain in the record. The unexecuted 256- and 1024-output proposals are preserved separately.

The fixed workload is the original eight 16407-token C8 requests, concurrency 8, three consecutive
batches per arm, Native first and BidKV second: 48 requests, 4096 forced output tokens per request,
at most 196608 output tokens. Temperature 0, seed 17 and all shared default optimizations remain as
in primary MOD v4. Both arms use the same ae127 scheduler carrier and b7a845 Ascend CPU topology
compatibility carrier. Only the BidKV policy switch differs between arms.

The original 60-minute clock starts before Native cold initialization. It includes both cold
initializations and every publication or metadata park. Generation clients stop at the original
deadline minus 15 seconds to reserve time for counters and record sealing. A client timeout is the
smaller of 600 seconds and the remaining generation window; it is a whole-request deadline. Owned
process cleanup has a separate recorded grace period. Missing or partial usage is never synthesized.
Each model is loaded once with fresh namespaces.

Zero-generation admission is the separately published 13dc packet, unchanged. The official worker
extension and localhost dev RPC are common read-only observability changes on both arms. They expose
native JSON geometry and configuration metadata, never real tensor values, repr, copies or pointers.
The extension changes neither configuration nor allocator state. Actual groups, pool capacity, page
sizes, descriptors, optimization flags, loaded module paths, worker affinities and eight exact input
token hashes must pass that owner.

`capacity_rules.py` operates only on the resulting JSON. The pinned official coordinator creates one
common BlockPool and adds group requirements; layer count is not multiplied into pool-block
requirements. One null block is reserved. Full attention uses ceiling(context / block size); aligned
Mamba first prefill uses 1 + speculative states and its steady bound is 2 + speculative states. The
report states actual descriptor bytes, group units, first-input admission counts and the
maximum-context bound. The test permits generation only if the actual 65536 context is admitted and
the fixed end-of-request upper bound at the admission count exceeds usable capacity with at least
two admitted requests. This is **pressure possible**, not a prediction of exact scheduling or proof
that preemption or BidKV will occur. Unsupported layouts or no pressure produce zero generation; no
on-site KV, context or output adjustment is allowed.

Native actual capacity and the final source/plan are published and independently read back before
`generation-public-release.json` is written. Its required fields are listed in
`execution-plan.json`. That same public release expressly pre-authorizes Candidate only if its
independently admitted geometry fingerprint is exactly equal to Native's, with the same go result.
Candidate does not wait for a second network publication: its actual confirmation and raw evidence
are preserved and published afterwards. This conditional rule replaces the earlier zero-generation
proposal's wording about two publication parks; all actual parks still consume the same original
clock. Any mismatch stops without Candidate generation. This rule does not authorize generation
without the initial actual-capacity release.

Each external request has a unique fixed-length explicit request ID and fresh salt. Observed SSE IDs
must equal `chatcmpl-` plus that ID; all 48 ID/salt mappings are independently checked at closure.
There is no external KV connector, and external successful restore counters must remain zero. APC
stays enabled. Unlike primary strict zero-reuse qualification, a preempted request may resume its
own resident prefix. All local prefix counters, usage cache details and preemption counters are
retained. Aggregate hits cannot establish per-hit self-resume provenance; any such explanation is an
inference, not a measured attribution. The pinned hash-salt chain and request ID source identities
are listed in `official-source-identities.json`.

`pressure_transport.py` is derived from the primary transport (SHA256
e372ac8bdc271550b380b67a7294350d6ffd266f363cdb23359e07b2c0e869f2), adding only consistent SSE
request-ID retention. Its default strict reuse gate remains enabled. Only this supplemental client
explicitly permits local prefix reuse. HTTP/SSE captures, partial records, complete accepted-token
usage and first-generated timings remain controlled originals. Public output contains numeric
records and hashes, not original prompts, responses or SSE.

The controller preserves actual client, API and admission-owner wait codes. There are no retries,
replacement rounds or selection of the best round. Incomplete batch throughput is null; successful
completed tokens are a lower bound. Mechanism claims require actual legal policy calls/selection and
no exception or invalid-selection fallback. The original selector's abstention guards and zero-call
outcomes remain visible. Even a positive paired synthetic result applies only to this artificial
configuration and single ordered cohort.

Official rules are pinned at vLLM commit
[0fc695f](https://github.com/vllm-project/vllm/tree/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665):
[shared pool and group sum](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/kv_cache_coordinator.py),
[Mamba allocation](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/single_type_kv_cache_manager.py),
[cache specs](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/kv_cache_interface.py),
[null block](https://github.com/vllm-project/vllm/blob/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665/vllm/v1/core/block_pool.py).
The exact local tracked bytes and Git blobs were checked independently on 81.
