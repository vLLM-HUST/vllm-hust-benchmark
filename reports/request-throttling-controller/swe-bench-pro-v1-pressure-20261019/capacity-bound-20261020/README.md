# Fixed-profile KV capacity analysis and proposed pressure profile

Status: **AWAITING_USER_DECISION; candidate not deployed**. This is a source/live-configuration
analysis, not a new hardware saturation experiment or performance result. No model restart or new
Agent batch occurred.

`report.json` retains frozen source anchors and SHA256 identities; private source bytes and live
endpoint receipts are not published. The public verifier checks arithmetic, consistency, and the
exact three proposed engine-field changes. It does not independently re-audit the private raw
evidence or prove an empirical saturation ceiling.

At max_num_seqs=16, regular standalone full-length request accounting yields 352 of 808 usable
blocks, or 43.564356%. The derivation assumes no KV connector, no speculative configuration, and the
recorded upstream allocation/free semantics; transient internal states and bugs are excluded. KV
memory is already reserved; block usage is not unallocated HBM.

The proposed separate shared pressure profile raises the engine execution ceiling and corresponding
graph capture limits to 256, preserving the original baseline and other engine settings. It still
uses 256 real Agent instances (8 official V1 tasks ×32 independent copies), time0 with no client
slots. It is neither 256 unique tasks nor trajectory replay. It does not guarantee 90% occupancy or
a performance gain. Deployment requires an explicit decision, a new shared profile version and fresh
matched controls through the existing RIDE workflow.

The original Feishu report was updated and read back at revision173. Existing failed 256-instance
measurements remain invalid for performance comparison and unchanged. No private task text,
solutions, patches, addresses, credentials, or token payloads are included.
